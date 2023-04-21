import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
# from ..adamwr import CyclicLRWithRestarts
import json
from loguru import logger

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")
            # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=5, t_mult=1.2, policy="cosine")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad()
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                # final loss
                tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(curr_epoch, iter_idx, num_iters)
            block2 = 'Time {:.2f} ({:.2f})'.format(batch_time.val, batch_time.avg)
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(losses_tracker['final_loss'].val, losses_tracker['final_loss'].avg)
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(key, value.val, value.avg)

            # print('\t'.join([block1, block2, block3, block4]))
            logger.info('\t'.join([block1, block2, block3, block4]))
    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def Write_TAL2_Json(video_id, obj):
    '''
    写入/追加json文件
    :param obj:
    :return:
    '''
    results = {}
    # if os.path.exists('labels.json'):
    with open('labels_submmision.json', 'r') as f:
        data = json.load(f)
        load_dict = data['results']
        #首先读取已有的json文件中的内容
        item_list = []
        if video_id in load_dict.keys():
            vid_obj_list = load_dict[video_id]
            num_item = len(vid_obj_list)
            for i in range(num_item):
                # verb      = vid_obj_list[i]['verb']
                noun      = vid_obj_list[i]['noun']
                action    = vid_obj_list[i]['action']
                score     = vid_obj_list[i]['score']
                segment   = vid_obj_list[i]['segment']
                # item_dict = {'verb':verb, 'noun':noun,'action':action, 'score':score, 'segment':segment}
                item_dict = {'noun':noun,'action':action, 'score':score, 'segment':segment}
                item_list.append(item_dict)
            item_list.append(obj) 
            load_dict.update({video_id:item_list})
        else:    
            print("new video_id has been done!")
            item_list.append(obj)
            load_dict.update({video_id:item_list})
            # pdb.set_trace()

        results = {"version": "0.2",
          "challenge": "action_detection",
          "sls_pt": -1,
          "sls_tl": -1,
          "sls_td": -1,
          "results": load_dict
        }
    with open('labels_submmision.json', 'w', encoding='utf-8') as f2:
        json.dump(data, f2, ensure_ascii=False)

import pdb
def Find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def segment_iou_(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[ 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[1] - candidate_segments[0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

#对特征提取模块所得到的lable,相邻的相同lable进行合并
#超过30S的相邻lable，只保留前30S即可
#lable合并后的prob取该段lable的最大值
def process_feature_lable(info):
    label_tmp = info.argsort(axis=1)
    lables = label_tmp[:,-1]
    probs = []
    for i in range(info.shape[0]):
        probs.append(info[i][lables[i]])
    print("origin feature info(top 10)",info[:20])
    print("labels")
    print(lables)
    print("probs")
    print(probs)
    n= len(lables)
    print("len(lables)", n)
    lable_new=[]
    time_new=[]
    score_new=[]
    prev_flag=[0]*16
    prev=lables[0]
    max_prob=probs[0]
    ts=0
    te=0
    t_interval_half=16/30
    # t_interval=32/30
    
    for i in range(1,n):
        cur = lables[i]
        
        if prev_flag[cur]:
            ts+=t_interval_half
            te+=t_interval_half
            continue
        else:
            prev_flag=[0]*16
        #相邻的相同标签进行合并
        if prev == cur:
            te+=t_interval_half
            max_prob = max(max_prob,probs[i])
            if te-ts>=30:
                lable_new.append(prev)
                time_new.append([ts,te])
                score_new.append(max_prob)
                prev_flag[prev]=1
                ts=te
                max_prob=0
        #标签不同，则将之前的合并标签进行存储
        else:
            te+=t_interval_half
            lable_new.append(prev)
            time_new.append([ts,te])
            score_new.append(max_prob)
            prev=cur
            ts=te
            max_prob=probs[i]
    #对最后一段进行存储
    te+=t_interval_half
    lable_new.append(prev)
    time_new.append([ts,te])
    score_new.append(max_prob)
    print("time_new",time_new)
    print("lable_new",lable_new)
    print("score_new",score_new)
    print("len(time_new)",len(time_new))
    # print(len(lable_new))
    return lable_new,time_new,score_new

def process_time_local_lable_(output,feature_lable,feature_time):
    num = len(feature_lable)
    time_lable = output['labels'][:num]
    time_seg = output["segments"][:num]
    matches=[]
    for i in range(num):
        cat = time_lable[i]
        max_tiou=0
        index=-1
        for j in range(num):
            if cat==feature_lable[j]:
                # tiou = segment_iou_(np.array(time_seg[i]),np.array(feature_time[j]))
                tiou = segment_iou_(np.array(feature_time[j]), np.array(time_seg[i]))
                if tiou>max_tiou:
                    max_tiou=tiou
                    index=j
        if index!=-1:
            matches.append([i,index])
    print("num of matches")
    print(len(matches))
    for item in matches:
        i, j = item[0],item[1]
        print(time_lable[i],feature_lable[j])
        print(time_seg[i],feature_time[j])

    return time_lable

#对时间定位模块所得的lable进行处理
#对每个lable，计算与特征提取模块的结果计算tiou,超过0.75的：
#若标签相同，则对得分进行加权；若标签不同，则对标签和得分进行替换
def process_time_local_lable(output,feature_lable,feature_time,feature_prob):
    num = len(feature_lable)
    time_lable = output['labels'][:num]
    time_seg = output["segments"][:num]
    time_score = output["scores"][:num]
    n = 0
    n1 = 0
    for i in range(num):
        cat = time_lable[i]
        for j in range(num):
            # tiou = segment_iou_(np.array(time_seg[i]),np.array(feature_time[j]))
            tiou = segment_iou_(np.array(feature_time[j]), np.array(time_seg[i]))
            if tiou > 0.9:
                n += 1
                # matches.append([i,j,tiou])
                if time_lable[i] != feature_lable[j]:
                    n1 += 1
                    print("Unmatched!!!!!!!!!!!!")
                    print("time_lable[i]", time_lable[i])
                    print("feature_lable[j]", feature_lable[j])
                    print("time_score[i]",time_score[i])
                    print("feature_prob[j]", feature_prob[j])
                    print("time_seg[i]", time_seg[i])
                    print("feature_time[j]", feature_time[j])
                    
                    time_lable[i] = feature_lable[j]
                    time_score[i] = torch.tensor(feature_prob[j])
                else:
                    # time_score[i] = feature_prob[j]*0.75+time_score[i]*0.25
                    time_score[i] = feature_prob[j]


    print("num of matches")
    print(n)
    print(n1)
    # for item in matches:
    #     i, j ,tiou = item[0], item[1], item[2]
    #     print(time_lable[i],feature_lable[j])
    #     print(time_seg[i],feature_time[j])
    #     if time_lable[i] != feature_lable[j]:
    #         time_lable[i] = feature_lable[j]
    #         time_score[i] = torch.tensor(feature_prob[j])
    #     else:
    #         time_score[i] = feature_prob[j]*0.75+time_score[i]*0.25



    return time_lable,time_score,time_seg

def process_feature_lable_v2(info):
    label_tmp = info.argsort(axis=1)
    lables    = label_tmp[:,-1]
    feats_num = len(lables)  #900
    probs     = np.zeros([feats_num],np.float32)
    for i in range(feats_num):
        probs[i] = info[i][lables[i]]
    

    feat_start_end_times       = np.zeros([feats_num, 2], np.float32)
    feat_start_end_times[:, 0] = np.arange(0, feats_num) * 16. / 30.
    feat_start_end_times[:, 1] = np.arange(2, feats_num + 2) * 16. / 30.
    prev = lables[0]
    same_label_start_indx = 0
    merge_labels = []
    lable_new    = []
    time_new     = []
    score_new    = []
    for i in range(1, feats_num):
        cur = lables[i]
        if cur != prev:
            ts   = feat_start_end_times[same_label_start_indx,0]
            te   = feat_start_end_times[i-1,1]
            prob = probs[same_label_start_indx:i].mean()

            merge_info = [prev, ts, te, prob]
            merge_labels.append(merge_info)
            same_label_start_indx = i
            prev = cur

            lable_new.append(prev)
            time_new.append([ts, te])
            score_new.append(prob)
    return lable_new, time_new, score_new


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }


    results_json = {}
    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):                     # video_list[0].keys:['video_id', 'feats', 'segments', 'labels', 'fps', 'duration', 'feat_stride', 'feat_num_frames']
        # pdb.set_trace()
        # forward the model (wo. grad)
        print(" ***************** eval video_id: ", video_list[0]['video_id'])
        with torch.no_grad():
            output = model(video_list)
            # upack the results into ANet format
            num_vids = len(output)
            # print("     num_vids: ",num_vids)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend([output[vid_idx]['video_id']] * output[vid_idx]['segments'].shape[0])
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

                # video_id = "P01_13"
                # obj = {"verb": 10,"noun": 7,"action": "10,7","score": 0.78582, "segment":[32.4444,40.25145]}
                # result = {}
                # Write_TAL2_Json(video_id, obj)

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP





# valid_one_epoch_with_cls_prob_submmition()
def valid_one_epoch_with_cls_prob_submmition(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    # assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()

    for iter_idx, video_list in (enumerate(val_loader, 0)):
        with torch.no_grad():
            output = model(video_list)
            # print("output")
            # print((output))

            # upack the results into ANet format
            num_vids = len(output)
            
            for vid_idx in range(num_vids):
                print("video_name", video_list[vid_idx]['video_id'])
                
                # lable_new, time_new, prob_new = process_feature_lable(video_list[vid_idx]['feats_probs']) 
                lable_new, time_new, prob_new = process_feature_lable_v2(video_list[vid_idx]['feats_probs']) 

                num_new = len(lable_new)
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend([output[vid_idx]['video_id']] * num_new)
                    #process_time_local_lable                   
                    time_lable, time_score, time_seg = process_time_local_lable(output[vid_idx], lable_new, time_new, prob_new)
                    results['t-start'].append(time_seg[:, 0])
                    results['t-end'].append(time_seg[:, 1])
                    results['label'].append(time_lable)
                    results['score'].append(time_score)
                    # results['t-start'].append(output[vid_idx]['segments'][:num_new][:, 0])
                    # results['t-end'].append(output[vid_idx]['segments'][:num_new][:, 1])
                    # results['label'].append(output[vid_idx]['labels'][:num_new])
                    # results['score'].append(output[vid_idx]['scores'][:num_new])

        # printing
        # if (iter_idx != 0) and iter_idx % (print_freq) == 0:
        #     # measure elapsed time (sync all kernels)
        #     torch.cuda.synchronize()
        #     batch_time.update((time.time() - start) / print_freq)
        #     start = time.time()

        #     # print timing
        #     print('Test: [{0:05d}/{1:05d}]\t'
        #           'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
        #           iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end']   = torch.cat(results['t-end']).numpy()
    results['label']   = torch.cat(results['label']).numpy()
    results['score']   = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
    # call the evaluator
    
    _, mAP = evaluator.evaluate(results, verbose=True)


    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

def valid_one_epoch_submmition(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }


    results_json = {}
    results_all  = {}
    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):                       # video_list[0].keys:['video_id', 'feats', 'segments', 'labels', 'fps', 'duration', 'feat_stride', 'feat_num_frames']
                                                                                # video_id: P01_11 feats: [2304, 1053] segments: [148, 2] labels: [148] fps: 30 duration: 561.6 feat_stride:16 feat_num_frames32
        # forward the model (wo. grad)

        # curr_cls_scores = np.asarray(cls_scores[vid])
        with torch.no_grad():
            output = model(video_list)                                          # output[0]: ['video_id', 'segments', 'scores', 'labels'] 
                                                                                                        # [400, 2]    [400]     [400]
            # upack the results into ANet format
            num_vids = len(output)
            # pdb.set_trace()
            # print("     num_vids: ",num_vids)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:                 # 2000
                    # pdb.set_trace()
                    results['video-id'].extend([output[vid_idx]['video_id']] * output[vid_idx]['segments'].shape[0])
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])
                    
                    # # # pdb.set_trace()
                    # for i in range(0,output[vid_idx]['segments'].shape[0]):
                    #     video_id = video_list[vid_idx]['video_id']
                    #     obj = {"noun": int(output[vid_idx]['labels'].numpy()[i]),"action": "","score": float(output[vid_idx]['scores'].numpy()[i]),
                    #          "segment":[float(output[vid_idx]['segments'][:, 0].numpy()[i]),float(output[vid_idx]['segments'][:, 1].numpy()[i])]}
                    #     # result = {}
                    #     # pdb.set_trace()
                        
                    #     Write_TAL2_Json(video_id, obj)

                #     # noun
                #     video_id = video_list[vid_idx]['video_id']
                #     obj_list = []
                #     for i in range(0,output[vid_idx]['segments'].shape[0]):
                #         obj = {"noun": int(output[vid_idx]['labels'].numpy()[i]),"action": "","score": float(output[vid_idx]['scores'].numpy()[i]),
                #                 "segment":[float(output[vid_idx]['segments'][:, 0].numpy()[i]),float(output[vid_idx]['segments'][:, 1].numpy()[i])]}
                #         obj_list.append(obj)
                #     video_dict = {video_id:obj_list}
                # results_all = dict(results_all, **video_dict)

                    # verb
                    video_id = video_list[vid_idx]['video_id']
                    obj_list = []
                    for i in range(0,output[vid_idx]['segments'].shape[0]):
                        obj = {"verb": int(output[vid_idx]['labels'].numpy()[i]),"action": "","score": float(output[vid_idx]['scores'].numpy()[i]),
                                "segment":[float(output[vid_idx]['segments'][:, 0].numpy()[i]),float(output[vid_idx]['segments'][:, 1].numpy()[i])]}
                        obj_list.append(obj)
                    video_dict = {video_id:obj_list}
                results_all = dict(results_all, **video_dict)
       
        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end']   = torch.cat(results['t-end']).numpy()
    results['label']   = torch.cat(results['label']).numpy()
    results['score']   = torch.cat(results['score']).numpy()
    
    results_json = {"version": "0.2",
          "challenge": "action_detection",
          "sls_pt": 2,
          "sls_tl": 3,
          "sls_td": 3,
          "results": results_all}
    
    # with open('labels_submmision_noun_test.json', 'w', encoding='utf-8') as f:
    #     json.dump(results_json, f, ensure_ascii=False) 
    
    with open('labels_submmision_verb_test.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False) 

    # pdb.set_trace()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file, 1024)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP