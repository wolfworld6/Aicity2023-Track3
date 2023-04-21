import os
from glob import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import video_transforms as video_transforms
import volume_transforms as volume_transforms
import modeling_feature
import argparse
from pathlib import Path
from timm.models import create_model
#import modeling_finetune
from torchvision import transforms
from transforms import *
from decord import VideoReader, cpu
from masking_generator import  TubeMaskingGenerator
from tqdm import tqdm
from kinetics import spatial_sampling, tensor_normalize




data_transform = video_transforms.Compose([
                video_transforms.Resize(256, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        self.crop_size = 224
        self.scale_size = 256
        self.new_length = args.num_frames
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.center_crop = GroupCenterCrop(self.crop_size)
        self.scale = GroupScale(self.scale_size)
        self.transform = transforms.Compose([                            
            self.scale,
            # self.center_crop,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

    def __call__(self, images):
        process_data , _ = self.transform(images)
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)
        return process_data


def get_args():
    parser = argparse.ArgumentParser('VideoMAE inference for video classification', add_help=False)

        # Dataset parameters
    parser.add_argument('--video_dir', default='/mnt/home/AIcity/croped_videos_A2', type=str,
                        help='inference video path')
    parser.add_argument('--save_dir', default='/mnt/home/aicitytrack3/tmp_test', type=str,
                        help='save path for videos feature')
    parser.add_argument('--model_path',default='./weights/hybrid_k700_vitl_rearview.pt', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--nb_classes', default=16, type=int,
                        help='number of the classification types')
    parser.add_argument('--view', default='Rear', type=str,choices=['Rear', 'Dash','Right'], help='views of videos to choose')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 32)
    parser.add_argument('--sampling_rate', type=int, default= 1)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'Track3','SSV2', 'UCF101', 'HMDB51','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')


    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=256)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=50, type=int)
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.,
    )

    # model = create_model(
    #     args.model,
    #     pretrained=False,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    #     decoder_depth=args.decoder_depth
    # )

    return model

def main(args):
    
    # clip_num = 32
    # over_lap = 16
    # skip_num = 1#bujangechouzhen ,30/fps
    clip_num = args.num_frames
    over_lap = 16
    skip_num = args.sampling_rate

    video_list = sorted(glob(os.path.join(args.video_dir, "*/{}*.mp4".format(args.view))))
    print(len(video_list))
    videos_t = video_list

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    print("get models")
    #patch_size = model.encoder.patch_embed.patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    #print(checkpoint)
    model.load_state_dict(checkpoint['module'])
    model.eval()
    
    #print(videos_t)
    for tmp_video in videos_t:
        print(tmp_video)
        input_dir, video_name = os.path.split(tmp_video)
        _, sub_dir = os.path.split(input_dir)
        save_sub_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(save_sub_dir):
            os.mkdir(save_sub_dir)
        out_file = os.path.join(save_sub_dir, video_name[:-4] + '.npz')
        if os.path.exists(out_file):
            print(out_file)
            print("file exists!")
            continue

        with open(tmp_video, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        duration = len(vr)
        print("duration:", duration)
        vr_num = (duration - clip_num + over_lap) // over_lap
        print("vr_num:", vr_num)
        feat_arr,v_arr,n_arr,a_arr, pred_arr = None,None,None,None,None
        for idx  in tqdm(range(vr_num)):
            strat_id = idx*over_lap
            end_id = idx*over_lap + clip_num*skip_num
            tmp = np.arange(strat_id, end_id, skip_num)
            frame_id_list = tmp.tolist()
            video_data = vr.get_batch(frame_id_list).asnumpy()
            #print(video_data.shape)
            imgs = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

            # transforms = DataAugmentationForVideoMAE(args)
            # imgs = transforms((imgs, None)) #C,T,H,W
            input_size = (args.input_size,args.input_size)
            trans_resize=video_transforms.Resize(size=input_size, interpolation='bilinear')
            imgs = trans_resize(imgs)
            imgs = torch.as_tensor(np.stack(imgs))
            #print(imgs.shape)
           
            imgs = tensor_normalize(imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #imgs = imgs.permute(0, 3, 1, 2)
            #print(imgs.shape)
            features = []
            with torch.no_grad():
                # scl, asp = (
                #     [0.08, 1.0],
                #     [0.75, 1.3333],
                # )
                # for id in [0, 1, 2]:
                # aug_imgs, _ = video_transforms.random_short_side_scale_jitter(imgs, 224, 224)
                # aug_imgs, _ = video_transforms.uniform_crop(aug_imgs, 224, id)
                # trans_resize = video_transforms.Resize((224,224), interpolation='bilinear')
                # aug_imgs, _ = trans_resize(imgs)             
                #print(imgs.shape)
                aug_imgs = imgs.permute(3, 0, 1, 2)
                aug_imgs = aug_imgs.unsqueeze(0)
                #print(aug_imgs.shape)
                aug_imgs = aug_imgs.to(device, non_blocking=True)
                feature, pred = model(aug_imgs)
                feature = feature.flatten()
                feat = feature.detach().cpu().numpy()

                pred = torch.softmax(pred, dim=1)
                pred = pred.flatten()
                pred = pred.detach().cpu().numpy()
                # features.extend(feature.detach().cpu().numpy())
                # pred = pred.detach().cpu().numpy()
                # cls = np.expand_dims(np.argmax(pred),axis=0)
                #feat = np.squeeze(feat)
            feat = np.array(feat)[None,...]
            pred = np.array(pred)[None,...]
            if feat_arr is None:
                feat_arr = feat
                pred_arr = pred
            else:
                feat_arr = np.concatenate((feat_arr, feat), axis=0)
                pred_arr = np.concatenate((pred_arr, pred), axis=0)

                #print("feat_arr_shape:", feat_arr.shape)
                #print("pred_arr_shape:", pred_arr.shape)
        #print("total_feat_arr_shape:", feat_arr.shape)
        np.savez( out_file, feats=feat_arr, pred=pred_arr)
        print(" extractor video Done.")

        # cap = cv2.VideoCapture(tmp_video)
        # len_frames = int(cap.get(7))
        # print(len_frames//16)
        # imgs_bs = []
        # feat_arr,v_arr,n_arr,a_arr = None,None,None,None
        # idid = 0
        # for i in range(len_frames):
        #     success, frame = cap.read()
        #     if not success: continue
        #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     imgs_bs.append(img_rgb)
        #     if (i+1)%16 ==0 and i>20:
        #         idid+=1
        #         print(idid)
        #         img_bs = np.stack(imgs_bs)
                
        #         # imgs_bs_torch = torch.as_tensor(np.stack(imgs_bs))
        #         # T H W C -> C T H W.
        #         imgs_bs_torch = imgs_bs_torch.permute(3, 0, 1, 2)
        #         imgs_bs = imgs_bs[(len(imgs_bs)-16):]
        #         # features = []
        #         with torch.no_grad():
        #             video_input = data_transform(imgs_bs)#todo:change
        #             video_input = torch.unsqueeze(video_input,dim=0).to(args.device)
        #             feature, pred = model(video_input)
        #             feature = feature.flatten()
        #             # features.extend(feature.detach().cpu().numpy())
        #             feat = feature.detach().cpu().numpy()
        #             cls = pred.detach().cpu().numpy()
        #             arr = np.array([feat, cls])
        #             if feat_arr is None:
        #                 feat_arr = arr
        #             else:
        #                 feat_arr = np.concatenate((feat_arr, arr), axis=0)
        #         print("feat_arr_shape:", feat_arr.shape)
        # print("totao_feat_arr_shape:", feat_arr.shape)
        # np.savez( out_file, feats=feat_arr)
        # print(" extractor video Done.")
        # cap.release()



if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
    
                
                




                

