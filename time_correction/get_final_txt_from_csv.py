import os 
import pandas as pd
from datetime import *
import csv
import numpy as np
import argparse


obj_path1 = "/mnt/home/dataset/AIcity2023-track3/video_ids.csv"
videoNames2Id={}
with open(obj_path1,'r',encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id = row['video_id']
        name = row['video_files']
        videoNames2Id[name]=id
        name1="Rear_view_user_id_"+name.split("id_")[1]
        name2="Right_side_window_user_id_"+name.split("id_")[1]
        videoNames2Id[name1]=id
        videoNames2Id[name2]=id
# print(videoNames2Id)

def get_dict_from_csv(csv_path):
    with open(csv_path,'r',encoding="utf-8") as csvfile:
        database_items={}
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = int(row['t-start'].split(".")[0])
            te = int(row['t-end'].split(".")[0])
            #将时间四舍五入取整
            if row['t-start'].split(".")[1][0]>='5':
                ts+=1
            if row['t-end'].split(".")[1][0]>='5':
                te+=1
            id = videoNames2Id[row['video-id']+".MP4"]
            label = row['label']
            if te-ts<1:
                continue
            if id not in database_items:
                database_items[id]={}
            if label not in database_items[id]:
                database_items[id][label]=[]

            item={}
            item['seg']=[ts,te]
            item['score']=row['score']
            item['angle']=row['video-id'].split('_')[0]
            # print(item['angle'])
            database_items[id][label].append(item)
        return database_items


def segment_iou_(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[ 0])
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

#只保留rear角度的结果,直接舍弃Dash,right
#对于相同video_id相同类别的预测结果，计算tiou,若大于阈值，保留得分高的
def reserve_rear_lable_by_tiou(infos,method,tiou_threshold):
    n=0
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)

            for i in range(num-1):
                #过滤Dash,right
                if items[i]['angle']!="Rear":
                    items[i]['score']="#"
                #过滤lable0
                if cat=="0":
                    items[i]['score']="#"

            for i in range(num-1):                
                if  items[i]['score']=="#":
                    continue
                for j in range(i+1,num):
                    if  items[j]['score']=="#":
                        continue
                    if  items[i]['score']=="#":
                        continue
                    target_segment = items[j]['seg']
                    candidate_segments = items[i]['seg']
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou>tiou_threshold:
                        n+=1
                        #两者都是Rear,保留得分高的
                        #比较到小数点后三位
                        if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                            items[j]['score']="#"
                        else:
                            items[i]['score']="#"
                        
    print("reserve num of rear lable:", n)

#只保留rear,Dash角度的结果,直接舍弃right
#对于相同video_id相同类别的预测结果，计算tiou,若大于阈值，保留得分高的
def reserve_rear_dash_lable_by_tiou(infos,method,tiou_threshold):
    n=0
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)

            for i in range(num-1):
                #过滤right
                if items[i]['angle']=="Right":
                    items[i]['score']="#"
                #过滤lable0
                if cat=="0":
                    items[i]['score']="#"

            for i in range(num-1):                
                if  items[i]['score']=="#":
                    continue
                for j in range(i+1,num):
                    if  items[j]['score']=="#":
                        continue
                    if  items[i]['score']=="#":
                        continue
                    target_segment = items[j]['seg']
                    candidate_segments = items[i]['seg']
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou>tiou_threshold:
                        n+=1

                        #比较到小数点后三位
                        if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                            items[j]['score']="#"
                        else:
                            items[i]['score']="#"
                        
    print("reserve num of rear and dash lable:", n)


#对于相同video_id相同类别的预测结果，计算tiou,
#对于tiou>设定阈值的：只保留rear角度的结果
def merge_rear_lable_by_tiou(infos,method,tiou_threshold):
    n=0
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            for i in range(num-1):
                #过滤lable0
                if cat=="0":
                    items[i]['score']="#"
                if  items[i]['score']=="#":
                    continue
                for j in range(i+1,num):
                    if  items[j]['score']=="#":
                        continue
                    if  items[i]['score']=="#":
                        continue
                    target_segment = items[j]['seg']
                    candidate_segments = items[i]['seg']
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou>tiou_threshold:
                        n+=1
                        #两者都是Rear,保留得分高的
                        if items[i]['angle']=="Rear" and items[j]['angle']=="Rear":
                            #比较到小数点后三位
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                            else:
                                items[i]['score']="#"
                        elif items[i]['angle']=="Rear":
                            items[j]['score']="#"
                        elif items[j]['angle']=="Rear":
                            items[i]['score']="#"
                        else:
                            #两者都不是Rear,保留得分高的
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                            else:
                                items[i]['score']="#"
    print("merge num of rear lable:", n)


#对于相同video_id相同类别的预测结果，计算tiou,
#对于tiou>设定阈值的：
# 1.比较两者的得分，舍弃掉得分低的结果
# 2.比较两者的时间范围，取min_start,max_end
# 3.比较两者的时间范围，取avg_start,avg_end
def merge_rear_dash_lable_by_tiou(infos,method,tiou_threshold):
    n=0
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            for i in range(num-1):
                #过滤lable0
                if cat=="0":
                    items[i]['score']="#"
                if  items[i]['score']=="#":
                    continue
                for j in range(i+1,num):
                    if  items[j]['score']=="#":
                        continue
                    if  items[i]['score']=="#":
                        continue
                    target_segment = items[j]['seg']
                    candidate_segments = items[i]['seg']
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou>tiou_threshold:
                        n+=1
                        #两者都是Rear,保留得分高的
                        if items[i]['angle']=="Rear" and items[j]['angle']=="Rear":
                            #比较到小数点后三位
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                            else:
                                items[i]['score']="#"
                        elif items[i]['angle']=="Rear":
                            items[j]['score']="#"
                        elif items[j]['angle']=="Rear":
                            items[i]['score']="#"
                        else:
                            #两者都不是Rear,保留Dash
                            #两者都是Dash,保留得分高的
                            if items[i]['angle']=="Dash" and items[j]['angle']=="Dash":
                                #比较到小数点后三位
                                if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                    items[j]['score']="#"
                                else:
                                    items[i]['score']="#"
                            elif items[i]['angle']=="Dash":
                                items[j]['score']="#"
                            elif items[j]['angle']=="Dash":
                                items[i]['score']="#"
                            else:
                                #两者都不是Dash,保留得分高的
                                if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                    items[j]['score']="#"
                                else:
                                    items[i]['score']="#"
                            

    print("merge num of rear and dash lable:", n)



#对于相同video_id相同类别的预测结果，计算tiou,
#对于tiou>设定阈值的：
# 1.比较两者的得分，舍弃掉得分低的结果
# 2.比较两者的时间范围，取min_start,max_end
# 3.比较两者的时间范围，取avg_start,avg_end
def merge_same_lable_by_tiou(infos,method,tiou_threshold):
    n=0
    for vid in infos:
        for cat in infos[vid]:
            items=infos[vid][cat]
            num = len(items)
            for i in range(num-1):
                #过滤lable0
                # if cat=="0":
                #     items[i]['score']="#"
                if  items[i]['score']=="#":
                    continue
                for j in range(i+1,num):
                    if  items[j]['score']=="#":
                        continue
                    if  items[i]['score']=="#":
                        continue
                    target_segment = items[j]['seg']
                    candidate_segments = items[i]['seg']
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou>tiou_threshold:
                        n+=1
                        if method == 1:
                            #比较到小数点后三位
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                            else:
                                items[i]['score']="#"
                            
                        elif method == 2:
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                                st = min(candidate_segments[0],target_segment[0])
                                et = max(candidate_segments[1],target_segment[1])
                                items[i]['seg'] = [st,et]
                            else:
                                items[i]['score']="#"
                                st = min(candidate_segments[0],target_segment[0])
                                et = max(candidate_segments[1],target_segment[1])
                                items[j]['seg'] = [st,et]
                            
                        elif method == 3:
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                                st = (candidate_segments[0]+target_segment[0])/2
                                et = (candidate_segments[1]+target_segment[1])/2
                                st = round(st,0)
                                et = round(et,0)
                                items[i]['seg'] = [int(st),int(et)]
                            else:
                                items[i]['score']="#"
                                st = (candidate_segments[0]+target_segment[0])/2
                                et = (candidate_segments[1]+target_segment[1])/2
                                st = round(st,0)
                                et = round(et,0)
                                items[j]['seg'] = [int(st),int(et)]                
                        elif method == 4:
                            if int(items[i]['score'].split('.')[1][0:3]) >= int(items[j]['score'].split('.')[1][0:3]):
                                items[j]['score']="#"
                                st = max(candidate_segments[0],target_segment[0])
                                et = min(candidate_segments[1],target_segment[1])
                                items[i]['seg'] = [st,et]
                            else:
                                items[i]['score']="#"
                                st = max(candidate_segments[0],target_segment[0])
                                et = min(candidate_segments[1],target_segment[1])
                                items[j]['seg'] = [st,et]

                        else:
                             print("method error!")
                             break
    print("merge num of same lable:", n)


def delete_diff_lable_by_tiou(infos,tiou_threshold):
    n=0
    for vid in infos:
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                for cat2 in infos[vid]:
                    if cat1!=cat2:
                        items2=infos[vid][cat2]
                        num2 = len(items2)
                        for j in range(num2):
                            if  items2[j]['score']=="#":
                                continue
                            if  items1[i]['score']=="#":
                                continue                 
                            target_segment = items2[j]['seg']
                            candidate_segments = items1[i]['seg']
                            tiou = segment_iou_(target_segment, candidate_segments)
                            if tiou>tiou_threshold:
                                # print(items[i]['score'])
                                #比较到小数点后三位
                                if int(items1[i]['score'].split('.')[1][0:3]) >= int(items2[j]['score'].split('.')[1][0:3]):
                                    if num1<num2 and num2>1:
                                        items2[j]['score']="#"
                                        n+=1
                                else:
                                    if num1>num2 and num1>1:
                                        items1[i]['score']="#"
                                        n+=1
                                
    print("delete num of diff lable:", n)


def get_top_n_per_vid(infos,top_n):
    n=0
    scores={}
    for vid in infos:
        scores[vid]=[]
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]

            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                #比较到小数点后三位
                scores[vid].append(int(items1[i]['score'].split('.')[1][0:3]))

    standard={}
    for vid in scores:
        v_scroe=scores[vid]
        if len(v_scroe)<=20:
            continue
        else:           
            v_scroe.sort(reverse=True)
            standard[vid]=v_scroe[top_n-1]

    for vid in infos:
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                if vid in standard and int(items1[i]['score'].split('.')[1][0:3])<standard[vid]:
                    items1[i]['score']="#"
                    n+=1            
                                
    print("delete num for top_n:", n)


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
                scores[vid][cat1].append(int(items1[i]['score'].split('.')[1][0:3]))
    
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
                if cat1 in standard[vid] and int(items1[i]['score'].split('.')[1][0:3])<standard[vid][cat1]:
                    items1[i]['score']="#"
                    n+=1            
                                
    print("delete num for top_1 for lable:", n)


def get_top_n_per_lable(infos,top_n):
    n=0
    scores={}
    top2_scores=[]
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
                scores[vid][cat1].append(int(items1[i]['score'].split('.')[1][0:3]))
    
    standard={}
    for vid in scores:
        standard[vid]={}
        for cat in scores[vid]:
            v_scroe=scores[vid][cat]
            if len(v_scroe)<=top_n:
                continue
            else:           
                v_scroe.sort(reverse=True)
                standard[vid][cat]=v_scroe[top_n-1]
                # top2_scores.append(v_scroe[top_n-1])
    
    
    # print("len(top2_scores")        
    # print(len(top2_scores))
    # top2_scores.sort(reverse=False)
    # print("top2_scores",top2_scores)
    # avg = sum(top2_scores)/len(top2_scores)

    for vid in infos:
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                if cat1 in standard[vid] and int(items1[i]['score'].split('.')[1][0:3])<standard[vid][cat1]:
                    items1[i]['score']="#"
                    n+=1  

                # elif cat1 in standard[vid] and int(items1[i]['score'].split('.')[1][0:3])==standard[vid][cat1] and standard[vid][cat1]<avg:
                #     items1[i]['score']="#"
                #     n+=1    

    print("delete num for top_n for lable:", n)


def filter_top_2_by_score(infos):
    n=0
    top2_info = []
    top2_scores=[]
    for vid in infos:     
        for cat1 in infos[vid]:
            items1=infos[vid][cat1]              
            num1 = len(items1)
            valid_num=0
            for i in range(num1):
                if  items1[i]['score']=="#":
                    continue
                else:
                    valid_num+=1
            if valid_num>1:
                top2_info.append([vid,cat1])

    for x in top2_info:
        vid,cat = x[0],x[1]
        items1=infos[vid][cat]              
        num1 = len(items1)
        tmp_score=999
        for i in range(num1):
            if  items1[i]['score']=="#":
                continue
            else:
                tmp_score = min(tmp_score,int(items1[i]['score'].split('.')[1][0:3]))
        top2_scores.append(tmp_score)

    print("top2_info",top2_info)
    print("top2_scores",top2_scores)
    top2_scores.sort(reverse=True)
    # print("top2_info",top2_info)
    print(" sort top2_scores",top2_scores)
    # filter_num = len(top2_scores)//2
    # #top2保留一半
    # standard = top2_scores[filter_num]
    #top2保留前3个得分高的
    standard = top2_scores[3]

    for x in top2_info:
        vid,cat = x[0],x[1]
        items1=infos[vid][cat]              
        num1 = len(items1)
        tmp_score=999
        tmp_idx=0
        for i in range(num1):
            if  items1[i]['score']=="#":
                continue
            else:              
                tmp = int(items1[i]['score'].split('.')[1][0:3])
                if tmp<tmp_score:
                    tmp_score=tmp
                    tmp_idx=i
        if tmp_score<=standard:
            print("tmp",tmp_score)
            items1[tmp_idx]['score']="#"
            n+=1
                                         
    print("filter num for top_2 by score:", n)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time correction for one model result')
    
    parser.add_argument('--csv_path', metavar='path', required=True,
                        help='the path to the csv file ')   
    parser.add_argument('--out_file', metavar='path', required=True,
                        help='the file name of the generated txt file')

    parser.add_argument('--merge_tiou_threshold', required=False, type=float, default=0.1, help='the tiou_threshold when merging results of same lable')
    parser.add_argument('--method', required=False, type=int, default=1, help='(1,2,3,4)')
        #对于tiou>设定阈值的：method:
    # 1.比较两者的得分，舍弃掉得分低的结果
    # 2.比较两者的时间范围，取min_start,max_end
    # 3.比较两者的时间范围，取avg_start,avg_end
    # 4.比较两者的时间范围，取max_start,min_end


    args = parser.parse_args()

    #对于不同的lable,若tiou>设置阈值，则过滤掉得分低 并且 数目多的lable
    # delete_tiou_threshold=0.1
    # #score_threshold:0.10
    # score_threshold=10
    # top_n_per_vid = 20
    # top_n_per_lable = 2


    csv_path = args.csv_path
    out_file = args.out_file
    merge_tiou_threshold = args.merge_tiou_threshold
    method = args.method

    #是否在输出的txt中加入score信息
    add_score = False
    # add_score = True
    if add_score:
        out_path="add_score_txt_files/"+out_file
    else:
        out_path="txt_files/"+out_file
    f1=open(out_path,"w+")

    infos = get_dict_from_csv(csv_path)
    
    # merge_same_lable_by_tiou(infos,method,merge_tiou_threshold)
    # merge_rear_lable_by_tiou(infos,method,merge_tiou_threshold)
    # merge_rear_dash_lable_by_tiou(infos,method,merge_tiou_threshold)

    # reserve_rear_dash_lable_by_tiou(infos,method,merge_tiou_threshold)
    # reserve_rear_lable_by_tiou(infos,method,merge_tiou_threshold)

    # delete_diff_lable_by_tiou(infos,delete_tiou_threshold)

    # get_top_n_per_vid(infos,top_n_per_vid)

    #对每个标签取top1
    get_top_1_per_lable(infos)
    #相同vid相同标签根据tiou融合
    merge_same_lable_by_tiou(infos,method,merge_tiou_threshold)


    # #1.对每个标签取top2
    # get_top_n_per_lable(infos,top_n_per_lable)
    # #2.相同vid相同标签根据tiou融合
    # merge_same_lable_by_tiou(infos,method,merge_tiou_threshold)
    # #3.相同vid不同标签根据tiou融合
    # delete_diff_lable_by_tiou(infos,delete_tiou_threshold)
    # #4.过滤掉top2中得分低的结果
    # filter_top_2_by_score(infos)
    

    
            
    n1=0
    n2=0
    id_nums=[0]*10
    for video_id in range(1,11):
        #去掉标签0
        for i in range(1,16):
        # for i in range(0,16):
            if str(i) in infos[str(video_id)]:
                item=infos[str(video_id)][str(i)]
                num = len(item)
                for j in range(num):
                    if item[j]['score']=="#":
                        n1+=1
                        continue       
                    if int(item[j]['score'].split('.')[1][0:2]) < score_threshold:
                        n2+=1
                        continue        

                    if add_score:     
                        f1.write(str(video_id)+" "+str(i)+" "+str(item[j]['seg'][0])+" "+str(item[j]['seg'][1])+" "+item[j]['score']+"\n")
                    else:
                        f1.write(str(video_id)+" "+str(i)+" "+str(item[j]['seg'][0])+" "+str(item[j]['seg'][1])+"\n")
                    
                    id_nums[video_id-1]+=1
    print("filter num by lable" ,n1)
    print("filter num by score", n2)
    print("nums of 10 videos", id_nums)





