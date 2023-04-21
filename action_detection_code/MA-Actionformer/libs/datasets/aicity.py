import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
import ipdb
import random
from loguru import logger

@register_dataset("aicity")
class AICityDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        feats_concat,
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)                             # aicity_track3.json
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        # pdb.set_trace()
        self.data_list = dict_db
        self.label_dict = label_dict
        self.feats_concat = feats_concat

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'aicity',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            # 'tiou_thresholds': np.linspace(0.8, 0.9, 3),
            # 'tiou_thresholds': np.linspace(0.5, 0.9, 5),
            'empty_label_ids': empty_label_ids
        }

    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        # if label_dict == num_classes:
        #     return []
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # pdb.set_trace()
        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    # pdb.set_trace()
                    label_dict[act['label']] = act['label_id']             # all label id

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # print("confirm json subset: {} =? split: {}.".format(value['subset'].lower(), self.split))
            # pdb.set_trace()
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # get fps if available
            if self.default_fps is not None:                              # 30                            
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        # load features
        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] + self.file_ext)
        # print("feats filename is {} .".format(filename))
        
        with np.load(filename) as data:
            # with np.load(filename, allow_pickle=True) as data:
            # logger.info("data keys : ", data.files)
            feats = data['feats'].astype(np.float32)
            if not self.is_training:
                if "videomae" in self.feat_folder:
                    feats_probs = data['a_prob']
                elif "SingleTrained" in self.feat_folder:
                    feats_probs = None
                else:
                    feats_probs = data['pred']    # ego4d
        # # logger.info("load feat1 shape [0]: {} [1]: {}.".format(feats.shape[0], feats.shape[1]))
        # if self.feats_concat:
        #     with np.load(os.path.join('./data/aicity/trained_features/features_ego4d_verb_vitl_track3_crop_pred_A1A2', video_item['id']+'.npz')) as data2:
        #         feats_mae = data2['feats'].astype(np.float32)
        #         feats_mae = F.interpolate(torch.from_numpy(feats_mae).permute(1, 0).unsqueeze(0), size=feats.shape[0], mode='linear', align_corners=False)[0,...].permute(1,0)
        #         # logger.info("load feat2 shape [0]: {} [1]: {}.".format(feats_mae.shape[0], feats_mae.shape[1]))
        #         feats_mae = feats_mae.numpy()
        #         feats     = feats_mae
        #     # feats[:, 0:1024] = feats_mae[:, 2048:]    
        # # logger.info("load feat_cat shape [0]: {} [1]: {}.".format(feats.shape[0], feats.shape[1]))
        # # logger.info("self.downsample_rate {} ".format(self.downsample_rate))
        # # pdb.set_trace()
        # # deal with downsampling (= increased feat stride)
        # # logger.info("load before feats shape [0]: {} [1]: {}.".format(feats.shape[0], feats.shape[1]))
   
        # if self.is_training:
        #     crop_feats, crop_video_item = self._TemporalRandomCrop(feats, video_item)
        #     # print(video_item.keys(), crop_video_item.keys())
        #     combined_keys = video_item.keys() | crop_video_item.keys()                                    # {'fps', 'segments', 'duration', 'labels', 'id'}
        #     video_item_comb = {}
        #     feats_aug_segs_list = []
        #     feats_aug_labs_list = []
        #     for key in combined_keys:
        #         if key == "segments":
        #             feats_aug_segs_list.append(video_item.get(key, []))
        #             feats_aug_segs_list.append(crop_video_item.get(key, []))
        #             # video_item_comb = {key: feats_aug_segs_list}
        #             video_item_comb.update({key: feats_aug_segs_list}) 
        #         elif key == "labels":
        #             feats_aug_labs_list.append(video_item.get(key, []))
        #             feats_aug_labs_list.append(crop_video_item.get(key, []))
        #             video_item_comb.update({key: feats_aug_labs_list}) 
        #         else:
        #             video_item_comb.update({key: video_item.get(key, [])})
        #         # video_item_comb = {key: video_item.get(key, []) + crop_video_item.get(key, [])}
        #     logger.info(video_item['duration'])
        #     logger.info(crop_video_item['duration'])
        #     print("*******************************************")
        #     logger.info(video_item_comb['duration'])
        #     # ipdb.set_trace()
        #     aug_feats = np.concatenate((feats, crop_feats), axis=0)
        
 
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
       
        # print("video_item: \n", video_item)                                                                                     # {'id': 'Rear_view_user_id_96715_NoAudio_5', 'fps': 30, 'duration': 512.0, 'segments': array([[  7.,  25.]]) \
                                                                                                                                    # 'labels': array([14,  2, 13, 13, 13, 10, 13,  3,  7,  7,  5,  8,  5, 12, 11,  1, 14, 9,  4,  0, 15])}                 

        if video_item['segments'] is not None:
            segments = torch.from_numpy((video_item['segments'] * video_item['fps']- 0.5 * self.num_frames) / feat_stride)      # 
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        if self.is_training:
            # return a data dict
            data_dict = {'video_id'        : video_item['id'],
                         'feats'           : feats,      # C x T
                         'segments'        : segments,   # N x 2
                         'labels'          : labels,     # N
                         'fps'             : video_item['fps'],
                         'duration'        : video_item['duration'],
                         'feat_stride'     : feat_stride,
                         'feat_num_frames' : self.num_frames}
        else:
            data_dict = {'video_id'        : video_item['id'],
                         'feats'           : feats,         # C x T
                         'feats_probs'     : feats_probs,   # C x numclass
                         'segments'        : segments,      # N x 2
                         'labels'          : labels,        # N
                         'fps'             : video_item['fps'],
                         'duration'        : video_item['duration'],
                         'feat_stride'     : feat_stride,
                         'feat_num_frames' : self.num_frames}
        # data_dict = {'video_id'        : video_item['id'],
        #              'feats'           : feats,      # C x T
        #              'segments'        : segments,   # N x 2
        #              'labels'          : labels,     # N
        #              'fps'             : video_item['fps'],
        #              'duration'        : video_item['duration'],
        #              'feat_stride'     : feat_stride,
        #              'feat_num_frames' : self.num_frames}
        # print("feats mode is {}.".format(self.is_training))
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict

    def _TemporalRandomCrop(self, raw_feature, video_item, stride=0.5, clip_length=4, fps=30):
        """Random Crop the video along the temporal dimension.

        Required keys are:
        "duration_frame", "duration_second", 
        "feature_frame","annotations", 
        """
        
        self.crop_length = 256
        video_second = video_item['duration']
        feature_length = raw_feature.shape[0] # Total features extract from full video
        segments = video_item['segments']
        labels = video_item['labels']
        n_segments = len(segments)

        # convert crop_length to second unit
        haft_clip_length = clip_length / 2
        patch_length = min(self.crop_length, feature_length)
        patch_length_second = (patch_length) * stride
        # logger.info("feat_len {} segments len {} video_second {} patch_length_second {}".format(feature_length, n_segments, video_second, patch_length_second))

        # patch_length_second = (patch_length-1)*stride+clip_length
        start_max = video_second - patch_length_second

        while True:
            # Select the start frame randomly
            i = random.randint(0, n_segments - 2)
            start_0 = 0 if i == 0 else segments[i - 1][1]
            start_1 = min(segments[i][0], start_max)
            start_choice = np.arange(start_0,start_1,stride).tolist()
            if len(start_choice) == 0:
                continue
            start = random.choice(start_choice)
            end   = start + patch_length_second

            # Crop the feature according to the start and end frame
            start_feature_idx = int((start + 1) / stride)
            end_feature_idx = start_feature_idx + patch_length
            raw_feature = raw_feature[start_feature_idx:end_feature_idx, :]
            
            # Modify the labels
            new_segments, new_labels = [], []
            for _seg, _label in zip(segments, labels):
                if _seg[0] >= start and _seg[0] < end:
                    # _start=_seg[0]-(start+stride),
                    # _end= min(_seg[1]-(start+stride),patch_length_second)
                    _start = _seg[0] - start
                    _end   = min(_seg[1] - start, patch_length_second)
                    new_segments.append([_start, _end])
                    new_labels.append(_label)
            video_item['segments'] = np.array(new_segments)
            video_item['labels']   = np.array(new_labels)
            video_item['duration'] = patch_length_second
            return raw_feature, video_item