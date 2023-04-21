# -*- coding: utf-8 -*-


import os
import argparse 
import glob 
import cv2
import torch
from utils.torch_utils import select_device


def convert(xyxy):
    x1, y1 = xyxy[0], xyxy[1]
    w = int(xyxy[2]) - int(x1)
    h = int(xyxy[3]) - int(y1)
    
    return (x1,y1,w,h)


def init(video_path):
    cap = cv2.VideoCapture(video_path)
    full_rate =  rate = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return full_rate, width, height, vid_length


def compute_IoU(box1, box2, x1y1x2y2=True,
                GIoU=False, DIoU=False, 
                CIoU=False, eps=1e-7):

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)),0) * \
            max((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)),0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou  # Iou


def crop_save_driver_vid(vid_name,
                      vid_path,
                      max_xyxy_list,
                      full_rate,
                      out_file = '/workspace/noura/AI_city_challenge/depug/'):
   
    out_file_name = os.path.join(out_file, vid_name+'.mp4')
    x, y, width, height = convert(max_xyxy_list)
    
    print("saving video: "+ vid_name)
    print("width, hight", int(width),', ', int(height) )
    
    output = cv2.VideoWriter(out_file_name, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             full_rate, 
                             (int(width), int(height)))
    
    stream = cv2.VideoCapture(vid_path)
    counter = 0
    # while (1):
    #     ret, frame = stream.read()
    #     if not ret:
    #          break
    #     frame = (frame[int(max_xyxy_list[1]):int(max_xyxy_list[3]),
    #                    int(max_xyxy_list[0]):int(max_xyxy_list[2])])        
    #     output.write(frame)
    #     counter += 1
    #     if counter% 1000 == 0:
    #         print(counter, "frames has been saved")
    # print("video " , vid_name, " has been saved")
    # return

    frames=[]
    while (1):
        ret, frame = stream.read()
        if not ret:
             break           
        frames.append(frame)  
    print("len1")
    print(len(frames))
    for frame in frames:
        frame = (frame[int(max_xyxy_list[1]):int(max_xyxy_list[3]),
                       int(max_xyxy_list[0]):int(max_xyxy_list[2])])   
    for frame in frames:
        output.write(frame)
        counter += 1
        if counter% 1000 == 0:
            print(counter, "frames has been saved")
    print("video " , vid_name, " has been saved")
    return


car_id = 0
def driver_tube_construction( vid_name,
                                vid_path,
                                frame,
                                frame_id,
                                frame_predictions,
                                vehicles_tubes,
                                full_rate,
                                iou_threshhold =0.90,
                                miss_detection =7):
    miss_detection = 60
    global car_id 
    global saved 
    saved = 1
    indices_array = []

    for x1y1x2y2 in frame_predictions:
        if x1y1x2y2[5] !=0 or x1y1x2y2[4]<=0.50: ###class number 
            continue
        
        # flag for appending bbox == new car   False ==> not yet appended  True ==> appended
        append_flag = False

        # if there is pervious detected car check if the car is the same 
        if vehicles_tubes:
            for bbox_num in range(len(vehicles_tubes)):

                iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)
                if iou >= iou_threshhold:
                 
                    vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())
                    vehicles_tubes[bbox_num]['frame_id'].append(frame_id)
                    vehicles_tubes[bbox_num]['miss_detection'] = miss_detection

                    indices_array.append(bbox_num)
                        
                    append_flag = True
                    # end -- searching for matching car has been completed

        # append new detected car 
        if not append_flag:
            vehicles_tubes.append({'id': car_id,
                                  'xyxy_list':[x1y1x2y2.tolist()],
                                  'frame_id': [frame_id],
                                  'miss_detection':miss_detection})
            indices_array.append(len(vehicles_tubes)-1)
            car_id +=1
    # x1, y1 = xyxy[0], xyxy[1]
    # w = int(xyxy[2]) - int(x1)
    # h = int(xyxy[3]) - int(y1)
    # print(int(frame_predictions[-1][2]) - int(frame_predictions[-1][0]) , 
    #       int(frame_predictions[-1][3]) - int(frame_predictions[-1][1]))
        
    if len(frame_predictions) != 0 and ((int(frame_predictions[-1][2]) - int(frame_predictions[-1][0]) == 1503 and 
                                        int(frame_predictions[-1][3]) - int(frame_predictions[-1][1]) == 830) or
                                        (int(frame_predictions[-1][2]) - int(frame_predictions[-1][0]) == 1589 and
                                        int(frame_predictions[-1][3]) - int(frame_predictions[-1][1] == 987))):
    
        print("width, hight in exception", frame_predictions[-1][2] - frame_predictions[-1][0],
              frame_predictions[-1][3] - frame_predictions[-1][1])
        print ("vid name in exception",vid_name )
    # if vid_name == 'Rear_view_user_id_42271_NoAudio_4' or vid_name == 'Rear_view_User_id_65818_NoAudio_1' :
       
        for index in reversed(range(len(vehicles_tubes))):
            # if there is pervious detected car check if the car is the same 
            if vehicles_tubes:
                for bbox_num in range(len(vehicles_tubes)):
                    iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)
                    if iou >= iou_threshhold:
                        vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())

    return


def main(model, vid_name, video_path, weights, out_file):
    global saved 
    saved = 1
    full_rate, width, height, vid_length = init(video_path)
    
    count = 0
    print(vid_name, full_rate, width, height, vid_length )
    
    stream = cv2.VideoCapture(video_path)
    first_frame = True

    vehicles_tubes = [] 
    iou_threshhold = 0.30
    frame_id = 1
   
    # while (1):
        
    #     ret, frame = stream.read()
        
    #     if not ret:
    #          break

    #     #model infrence function 
    #     predictions = model(frame)
    #     #for each  frame
    #     for frame_predictions in predictions.xyxy:
    #         driver_tube_construction( vid_name,
    #                                     video_path,
    #                                     frame,
    #                                     frame_id,
    #                                     frame_predictions,
    #                                     vehicles_tubes,
    #                                     full_rate = full_rate,
    #                                     iou_threshhold = iou_threshhold)

    #     count +=1
    #     frame_id +=1
    frames=[]
    while (1):       
        ret, frame = stream.read()       
        if not ret:
             break
        frames.append(frame)  
    print("len2")
    print(len(frames))
    for frame in frames:

        #model infrence function 
        predictions = model(frame)
        #for each  frame
        for frame_predictions in predictions.xyxy:
            driver_tube_construction( vid_name,
                                        video_path,
                                        frame,
                                        frame_id,
                                        frame_predictions,
                                        vehicles_tubes,
                                        full_rate = full_rate,
                                        iou_threshhold = iou_threshhold)

        count +=1
        frame_id +=1


    max_area = 0
    max_xyxy_list = []
    car = 0
    for index in reversed(range(len(vehicles_tubes))):
        car = vehicles_tubes.pop(index)
        area = []
        for i in range(len(car['xyxy_list'])):
                x, y, width, hight = convert(car['xyxy_list'][i])
                area.append(width * hight)

        tmp = max(area)
        indexx = area.index(tmp)
        xyxy_list = car['xyxy_list'][indexx]
       
        if tmp > max_area:
              max_area = tmp
              max_xyxy_list = xyxy_list
              car = car
        saved = 0
    
    if saved == 0:         
        result = crop_save_driver_vid(vid_name,video_path, max_xyxy_list, full_rate, out_file)
        saved += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='driver tracking')
    
    parser.add_argument('--vid_path', metavar='path', required=False,
                        help='the path to the folder that contains video/videos in .mp4 format .. ')
    
    parser.add_argument('--out_file', metavar='path', required=False,
                        help='the path where the tracked driver videos will be saved ..')
    args = parser.parse_args()
    videos_path = glob.glob(args.vid_path+'/*/*')
    out_file = args.out_file+'/'
    
    
    weights ='/yolov5/yolov5s.pt'
    model = torch.hub.load('yolov5','yolov5s', pretrained=True, source='local', force_reload=True).autoshape()
    device = select_device(0)
    model = model.to(device)
        
    for video_path in videos_path:
        if video_path[-4:]==".MP4":
            car_id = 0
            vid_name = str(os.path.basename(video_path))[:-4]
            print("start processing video: ", vid_name)
            tmp_path = "user_id_"+vid_name.split("_")[-3]
            out_path = out_file+tmp_path
            print("out_path",out_path)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            main(model, vid_name, video_path, weights, out_path)

    # # video_path = "/mnt/home/dataset/AIcity2023-track3/A2/user_id_49989/Dashboard_user_id_49989_NoAudio_5.MP4"
    # # video_path = "/mnt/home/dataset/process_track3/Dashboard_user_id_49989_NoAudio_5.mp4"
    # video_path = "/mnt/home/dataset/AIcity2023-track3/A1/user_id_60167/Right_side_window_user_id_60167_NoAudio_5.MP4"
    # vid_name = str(os.path.basename(video_path))[:-4]
    # out_path = "/mnt/home/dataset/process_track3/croped_tmp"
    # main(model, vid_name, video_path, weights, out_path)


