#coding:utf-8
import pdb,os
import numpy as np

import pickle
import cv2
 

base_path = '/mnt/home/aicitytrack3/A1Feature/UniformerV2_clip400+k710+k700_A1/'
out_path  = '/mnt/home/aicitytrack3/A1Feature/UniformerV2_clip400+k710+k700_A1_3angle2one/'
subdirs = os.listdir(base_path)
for subdir in subdirs:
    videos_ = os.listdir(base_path+'/'+subdir)
    feat_owns = None
    videos_.sort()
    
    n_shape = 10000000
    for video_ in [videos_[0],videos_[2],videos_[4]]:
        print(video_)
        file_name = os.path.join(base_path, subdir,video_)
        feat_own= np.load(file_name) # 
        feat_own = feat_own['feats']
        n_shape = min(feat_own.shape[0],n_shape)
        feat_own = feat_own[:n_shape,1024:2048]
        if feat_owns is None:
            feat_owns = feat_own
        else:
            
            feat_owns = np.concatenate((feat_owns[:n_shape,:], feat_own), axis=1)

        # print(feat_owns.shape)
    out_path1 = out_path+'/'+subdir
    os.makedirs(out_path1, exist_ok=True)
    np.savez(out_path1+'/'+video_.split('user')[1], feats=feat_owns)
    
    # break
    n_shape = 1000000
    feat_owns = None
    for video_ in [videos_[1],videos_[3],videos_[5]]:
        print(video_)
        file_name = os.path.join(base_path, subdir,video_)
        feat_own= np.load(file_name) # 
        feat_own = feat_own['feats']
        n_shape = min(feat_own.shape[0],n_shape)
        feat_own = feat_own[:n_shape,1024:2048]
        if feat_owns is None:
            feat_owns = feat_own
        else:
            
            feat_owns = np.concatenate((feat_owns[:n_shape,:], feat_own), axis=1)

        # print(feat_owns.shape)
    out_path1 = out_path+'/'+subdir+'/'+video_.split('user')[1]
    os.makedirs(out_path, exist_ok=True)
    np.savez(out_path1 , feats=feat_owns)