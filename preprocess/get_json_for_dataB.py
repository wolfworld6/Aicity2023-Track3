#coding:utf-8
import pickle
import pandas as pd
import numpy as np
import pdb
from sys import argv
import csv
import os
import cv2
import random
import json
import argparse


shell = {
	"version":"AI CITY 2023 track-3",
	"database":{}
}


_MINUTES_TO_SECONDS=60
_HOURS_TO_SECONDS=3600
def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * _HOURS_TO_SECONDS + minutes * _MINUTES_TO_SECONDS + seconds
    return total_seconds


def load_anno_csv(obj_path):
    # json_save = open('test_timestep.json', 'w+', encoding='utf-8')
    count = 0
    video_list=[]
    with open(obj_path,'r',encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        annos=[]
        database_items={}
        cur_file_name=""
        # pre_file_name=""
        first_flag=True
        for row in reader:
            count += 1
            # print(row['Filename'])
            # print("reader csv count {} data {} !".format(count,row))
            obj = {"label": int(row['Label (Primary)'][6:]),"label_id": int(row['Label (Primary)'][6:]), "segment":[timestamp_to_seconds(row['Start Time']),
                timestamp_to_seconds(row['End Time'])]}

            if row['Filename']!="" and first_flag:
                database_items[row['Filename']]={}
                # pre_file_name = cur_file_name
                cur_file_name = row['Filename']    
                first_flag = False

                annos.append(obj)
            elif row['Filename']!="" and not first_flag:
                database_items[cur_file_name]["annotations"]=annos
                annos=[]
                annos.append(obj)
                video_list.append(database_items)
                database_items={}
                cur_file_name = row['Filename']  
                database_items[row['Filename']]={}
            else:
                annos.append(obj)



        database_items[cur_file_name]["annotations"]=annos
        # print("cur_file_name ",cur_file_name)
        # print("annos ",annos)
        video_list.append(database_items)

    return video_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    
    parser.add_argument('--data_path', metavar='path', required=True,
                        help='the path to the B dataset ')   
    parser.add_argument('--lable_path', metavar='path', required=True,
                        help='the path to the annotation files of B')
    parser.add_argument('--json_output', metavar='path', required=True,
                        help='the path to the generated json file')

    args = parser.parse_args()

    dir_path = args.data_path
    dir_path1 = args.lable_path
    json_output = args.json_output


    n=0
    for i in os.listdir(dir_path1):
        j = os.path.join(dir_path1,i)
        if j[-4:]==".csv":
            video_items = load_anno_csv(j)
            for p in range(len(video_items)):
                for q in video_items[p]:
                    tmp = q.split("_")
                    # print(tmp)
                    # n+=1
                    # print(n)
                    if tmp[0]=="Dashboard":
                        continue
                        video_name = "Dashboard"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                    elif tmp[0]=="Rearview":                           
                        video_name = "Rear_view"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                    elif tmp[0]=="Rear":                           
                        video_name = "Rear_view"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                    elif tmp[0]=="Right":
                        continue
                        video_name = "Right_side_window"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                    # print("video_name")
                    # print(video_name)
                    users_dir = os.path.join(dir_path,"user_id_"+tmp[-2])
                    video_path = os.path.join(users_dir,video_name)
                    cap = cv2.VideoCapture(video_path)
                    
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # print("video W:{},H:{},fps:{},total_num:{}".format(frame_w, frame_h, fps, frame_length))
                    video_time = frame_length/fps
                    
                    video_items[p][q]["resolution"] = str(frame_w)+"*"+str(frame_h)
                    video_items[p][q]["duration"] = video_time
                    video_items[p][q]["subset"] = "validation"

                    shell["database"][video_name[0:-4]]=video_items[p][q]

    with open(json_output, "w") as f:
        json.dump(shell, f, ensure_ascii=False, indent=4)




            







