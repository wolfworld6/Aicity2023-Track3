import os
import pandas as pd
import numpy as np
import argparse

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def segment_iou_(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[0])
    tt2 = np.minimum(target_segment[1], candidate_segments[1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[1] - candidate_segments[0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    if segments_union==0:
        return 0
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

#对于相同video_id相同类别的预测结果，计算tiou,
#将tiou>设定阈值的所有结果放入一个列表，一次性计算出所有结果的均值（非两两比较）
#对于tiou>设定阈值的所有结果：取avg_start,avg_end
def merge_same_lable_by_tiou_del_noTiou(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                #跳过被时间范围中位数或均值筛掉的结果
                if  items[i]['seg']==["#"]:
                    print("i",i)
                    continue
                candidate_segments=[items[i]['seg']]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    #跳过被时间范围中位数或均值筛掉的结果
                    if  items[j]['seg']==["#"]:
                        print('j',j)
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        candidate_segments.append(items[j]['seg'])
                        visited[j]=1
                visited[i]=1

                #舍弃掉无tiou交集的结果
                if len(candidate_segments)==1:
                    continue
                st=0
                et=0
                for seg in candidate_segments:
                    st+=seg[0]
                    et+=seg[1]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                item['seg'] = [int(st/len(candidate_segments)),int(et/len(candidate_segments))]
                new_info[vid][cat].append(item)
    return new_info

def merge_same_lable_by_tiou_check_time(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                candidate_segments=[items[i]['seg']]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        if abs(segment1[0]-segment2[0])+abs(segment1[1]-segment2[1])<20:
                            candidate_segments.append(items[j]['seg'])
                            visited[j]=1
                visited[i]=1

                st=0
                et=0
                for seg in candidate_segments:
                    st+=seg[0]
                    et+=seg[1]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                item['seg'] = [int(st/len(candidate_segments)),int(et/len(candidate_segments))]
                new_info[vid][cat].append(item)
    return new_info



def merge_same_lable_by_tiou(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                candidate_segments=[items[i]['seg']]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        candidate_segments.append(items[j]['seg'])
                        visited[j]=1
                visited[i]=1

                st=0
                et=0
                for seg in candidate_segments:
                    st+=seg[0]
                    et+=seg[1]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                item['seg'] = [int(st/len(candidate_segments)),int(et/len(candidate_segments))]
                new_info[vid][cat].append(item)
    return new_info


def merge_same_lable_by_tiou_add_score_del_noTiou(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                candidate_segments=[[items[i]['seg'][0],items[i]['seg'][1],items[i]['score']]]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        candidate_segments.append([items[j]['seg'][0],items[j]['seg'][1],items[j]['score']])
                        visited[j]=1
                visited[i]=1

                #舍弃掉无tiou交集的结果
                if len(candidate_segments)==1:
                    continue
                
                st=0
                et=0
                sc=0
                for seg in candidate_segments:
                    st+=seg[0]
                    et+=seg[1]
                    sc+=seg[2]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                fenMu = len(candidate_segments)
                item['seg'] = [int(st/fenMu),int(et/fenMu)]
                item['score'] = sc/fenMu
                new_info[vid][cat].append(item)
    return new_info

def merge_same_lable_by_tiou_add_score(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                candidate_segments=[[items[i]['seg'][0],items[i]['seg'][1],items[i]['score']]]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        candidate_segments.append([items[j]['seg'][0],items[j]['seg'][1],items[j]['score']])
                        visited[j]=1
                visited[i]=1
                
                st=0
                et=0
                sc=0
                for seg in candidate_segments:
                    st+=seg[0]
                    et+=seg[1]
                    sc+=seg[2]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                fenMu = len(candidate_segments)
                item['seg'] = [int(st/fenMu),int(et/fenMu)]
                item['score'] = sc/fenMu
                new_info[vid][cat].append(item)
    return new_info

#对于相同video_id相同类别的预测结果，计算tiou,
#将tiou>设定阈值的所有结果放入一个列表，一次性计算出所有结果的均值（非两两比较）
#对于tiou>设定阈值的所有结果：start取(score1*start1+score2*start2+...)/(score1+score2+...)
def merge_same_lable_by_tiou_and_score(infos,tiou_threshold):
    new_info={}
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            visited=[0]*num
            
            for i in range(num-1):
                #跳过已经匹配到的
                if visited[i]:
                    continue
                candidate_segments=[[items[i]['seg'][0],items[i]['seg'][1],items[i]['score']]]
                for j in range(i+1,num):
                    if visited[j]:
                        continue
                    segment1 = items[j]['seg']
                    segment2 = items[i]['seg']
                    tiou = segment_iou_(items[i]['seg'], items[j]['seg'])
                    if tiou>tiou_threshold:
                        candidate_segments.append([items[j]['seg'][0],items[j]['seg'][1],items[j]['score']])
                        visited[j]=1
                visited[i]=1

                #舍弃掉无tiou交集的结果
                if len(candidate_segments)==1:
                    continue
                st=0
                et=0
                weights=0
                for seg in candidate_segments:
                    st+=seg[0]*seg[2]
                    et+=seg[1]*seg[2]
                    weights+=seg[2]
                if vid not in new_info:
                    new_info[vid]={}
                if cat not in new_info[vid]:
                    new_info[vid][cat]=[]
                item={}
                item['seg'] = [int(st/weights),int(et/weights)]
                new_info[vid][cat].append(item)
    return new_info


def get_top_1_per_lable(infos):
    n=0
    scores={}
    for vid in infos:
        scores[vid]={}
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            if cat1 not in scores[vid]:
                scores[vid][cat1]=[]
            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                scores[vid][cat1].append(items1[i]['score'])
    
    standard={}
    for vid in scores:
        standard[vid]={}
        for cat in scores[vid]:
            v_scroe=scores[vid][cat]
            if len(v_scroe)<=1:
                continue
            else:           
                v_scroe.sort(reverse=True)
                standard[vid][cat]=v_scroe[0]

    for vid in infos:
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                if cat1 in standard[vid] and items1[i]['score']<standard[vid][cat1]:
                    items1[i]['score']="#"
                    n+=1            
                                
    print("delete num for top_1 for lable:", n)


def filter_lable_by_avg_score(infos,standard,margin):
    n=0
    intervals={}
    for vid in infos:
        intervals[vid]={}
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            if cat1 not in intervals[vid]:
                intervals[vid][cat1]=[]
            num1 = len(items1)
            for i in range(num1):
                intervals[vid][cat1].append(items1[i]['seg'][1]-items1[i]['seg'][0])
    
    standard1={}
    if standard==1:     
        for vid in intervals:
            standard1[vid]={}
            for cat in intervals[vid]:
                v_scroe=intervals[vid][cat]
                if len(v_scroe)<=1:
                    continue
                else:           
                    v_scroe.sort(reverse=True)
                    mid_n=len(v_scroe)//2
                    #中位数
                    standard1[vid][cat]=v_scroe[mid_n]
    elif standard==2:
        for vid in intervals:
            standard1[vid]={}
            for cat in intervals[vid]:
                v_scroe=intervals[vid][cat]
                if len(v_scroe)<=1:
                    continue
                else:           
                    #平均数
                    tol=0
                    for x in v_scroe:
                        tol+=x
                    standard1[vid][cat]=tol/len(v_scroe)
    else:
        return False

    
    for vid in infos:
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                seg_tmp = items1[i]['seg'][1]-items1[i]['seg'][0]
                if cat1 in standard1[vid]:
                    if seg_tmp<standard1[vid][cat1]-margin:
                        items1[i]['seg']=["#"]
                        n+=1  
                    elif seg_tmp>standard1[vid][cat1]+margin:
                        items1[i]['seg']=["#"]
                        n+=1          
                                
    print("delete num for over avg/mid time for lable:", n)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='time correction for multi model results')
    
    # parser.add_argument('--txt_path_list', type=str, required=True, help='the paths to the txt file ') 
    parser.add_argument('--txt_path_list', action='append', help='the paths to the txt file', required=True)  
    parser.add_argument('--out_file', metavar='path', required=True,
                        help='the file name of the merged txt file')

    parser.add_argument('--merge_tiou_threshold', required=False, type=float, default=0.01, help='the tiou_threshold when merging results of same lable')


    args = parser.parse_args()
    txt_list = args.txt_path_list
    print("txt_list",txt_list)
    out_txt = args.out_file
    merge_tiou_threshold = args.merge_tiou_threshold

    #standard=1:中位数
    #standard=2:平均数
    # standard=1
    # margin=5
    # out_txt = "txt_files/ehh_4_14.txt"
    # txt_list=[
    #     "/mnt/home/dataset/process_track3/txt_files/ego_4_14.txt",
    #     "/mnt/home/dataset/process_track3/txt_files/hyb_4_14.txt",
    #     "/mnt/home/dataset/process_track3/txt_files/hug_4_14.txt"
    # ]


    infos={}  #infos[vid][cat]=[st,et]
    for txt_path in txt_list:
        with open(txt_path,"r") as txt_file:
            for line in txt_file.readlines():
                # print(line.strip().split(" "))
                vid, cat, st, et = line.strip().split(" ")
                # vid, cat, st, et, score = line.strip().split(" ")
                if vid not in infos:
                    infos[vid]={}
                if cat not in infos[vid]:
                    infos[vid][cat]=[]
                item={}
                item['seg'] = [int(st),int(et)]
                # tmp_s=score.split(".")[1]
                # tmp_l=len(tmp_s)
                # tmp_score=int(tmp_s)/pow(10,tmp_l)
                # item['score'] = tmp_score
                infos[vid][cat].append(item)
    
    # new_info = merge_same_lable_by_tiou(infos,merge_tiou_threshold)
    # new_info = merge_same_lable_by_tiou_check_time(infos,merge_tiou_threshold)
    # new_info = merge_same_lable_by_tiou_add_score(infos,merge_tiou_threshold)
    # new_info = merge_same_lable_by_tiou_add_score_del_noTiou(infos,merge_tiou_threshold)
    # get_top_1_per_lable(new_info)
    # filter_lable_by_avg_score(infos,standard,margin)
    new_info = merge_same_lable_by_tiou_del_noTiou(infos,merge_tiou_threshold)
    # new_info = merge_same_lable_by_tiou_and_score(infos,merge_tiou_threshold)

    n1=0
    n2=0
    num_over_30s=0
    num_over_time=0
    id_nums=[0]*10
    f1=open(out_txt,"w+")
    for video_id in range(1,11):
        #去掉标签0
        for i in range(1,16):
        # for i in range(0,16):
            if str(i) in new_info[str(video_id)]:
                item=new_info[str(video_id)][str(i)]
                num = len(item)
                for j in range(num):    
                    # if item[j]['score']=="#":
                    #     n1+=1
                    #     continue               
                    # f1.write(str(video_id)+" "+str(i)+" "+str(item[j]['seg'][0])+" "+str(item[j]['seg'][1])+" "+item[j]['score']+"\n")
                    
                    if item[j]['seg']==["#"]:
                        num_over_time+=1
                        continue
                    if item[j]['seg'][1]-item[j]['seg'][0]>30:
                        num_over_30s+=1
                        continue

                    f1.write(str(video_id)+" "+str(i)+" "+str(item[j]['seg'][0])+" "+str(item[j]['seg'][1])+"\n")
                    id_nums[video_id-1]+=1
    print("filter num by lable" ,n1)
    print("nums of 10 videos", id_nums)
    print("num_over_30s",num_over_30s)
    print("num_over_time",num_over_time)