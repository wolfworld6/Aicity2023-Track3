import os 
import sys

import cv2
import csv
import json
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import argparse
from modeling_feature import vit_huge_patch16_224
import utils
from slowfast.datasets import utils as utils_slow
from glob import glob


import argparse

parser = argparse.ArgumentParser(description="Video Inference For Feature Extraction")
parser.add_argument("--device", type=str, default="cuda:1", help="device select to run function")
parser.add_argument("--video_dir", type=str, default="/mnt/home/AIcity/croped_videos_A2", help="folder path of input videos")
parser.add_argument("--ckpt_pth", type=str, default="./weights/k400_vith_rearview.pt", help="path of model checkpoint used to get feature")
parser.add_argument("--select_view", type=str, default='Rear', help="view of videos selected for feature")
parser.add_argument("--output_dir", type=str, default='/videomae_vitHK400_rearview_Feature/', help="path to save extracted video features")
args = parser.parse_args()


model = vit_huge_patch16_224()
checkpoint = torch.load(args.ckpt_pth, map_location='cpu')

utils.load_state_dict(model, checkpoint['module'], prefix='')

model.to(args.device)
model.eval

video_list = sorted(glob(os.path.join(args.video_dir, "*/*.mp4")))
for video_path in video_list:
    if args.select_view not in video_path:
        continue
    print(video_path)
    
    cap = cv2.VideoCapture(video_path)
    len_frames = int(cap.get(7))
    print(len_frames//16)
    imgs_bs = []
    feat_arr,v_arr,n_arr,a_arr = None,None,None,None
    idid = 0
    for i in range(len_frames):
        success, frame = cap.read()
        if not success: continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgs_bs.append(img_rgb)


        if (i+1)%16 ==0 and i>20:
            idid+=1
            print(str(idid)+"/"+ str(len_frames//16))
            #print(len_frames//16)
            imgs_bs_torch = torch.as_tensor(np.stack(imgs_bs[::2])) #32帧抽取16帧
            # imgs = torch.as_tensor(np.stack(imgs))

            '''for trained model in data'''
            # mean,std = [0.45,0.45,0.45],[0.225,0.225,0.225]
            # for videomae
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            frames = imgs_bs_torch # T H W C
            frames = utils_slow.tensor_normalize(frames, mean, std)
            
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            imgs_bs = imgs_bs[(len(imgs_bs)-16):]

            with torch.no_grad():
                video_inputs = []

                frames = torch.nn.functional.interpolate(
                    frames,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                video_input = frames#

                video_input = torch.tensor(np.array(video_input)).cuda(args.device)
                video_input = torch.unsqueeze(video_input,dim=0).cuda(args.device)
                feature,prediction = model(video_input) #3,3806  ,3 × 1024
                feature = feature.flatten()
                prediction = F.softmax(prediction, dim=1) # #3,3806

                feat   = feature.detach().cpu().numpy()[None,...]
                a_prob = prediction.detach().cpu().numpy()

                if feat_arr is None:
                    feat_arr = feat
                    a_arr = a_prob
                else:
                    feat_arr = np.concatenate((feat_arr, feat), axis=0)
                    a_arr    = np.concatenate((a_arr, a_prob), axis=0)

    print(feat_arr.shape)
    print(a_arr.shape)
    print(len_frames//32)

    out_path = args.output_dir + video_path.split('/')[-2]
    os.makedirs(out_path, exist_ok=True)
    out_file = video_path.split('/')[-1].split('.')[0] + '.npz'
    print( os.path.join(out_path, out_file))
    np.savez( os.path.join(out_path, out_file), feats=feat_arr, a_prob=a_arr)
    print(" extractor video Done.")
    
    cap.release()