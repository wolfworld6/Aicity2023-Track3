import subprocess
import os
import csv
import argparse

_MINUTES_TO_SECONDS=60
_HOURS_TO_SECONDS=3600
def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * _HOURS_TO_SECONDS + minutes * _MINUTES_TO_SECONDS + seconds
    return total_seconds


def load_anno_csv(obj_path):
    count = 0
    video_list=[]
    with open(obj_path,'r',encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        annos=[]
        database_items={}
        cur_file_name=""
        first_flag=True
        for row in reader:
            count += 1
            # print("reader csv count {} data {} !".format(count,row))
            obj = {"label": int(row['Label (Primary)'][6:]),"label_id": int(row['Label (Primary)'][6:]), "segment":[(row['Start Time']),
                (row['End Time'])]}
            #判断
            if timestamp_to_seconds(row['Start Time'])>=timestamp_to_seconds(row['End Time']):
                print("error!!!!!!!!!!!!!!")
                return False
            
                       
            if row['Filename']!="" and first_flag:
                database_items[row['Filename']]={}
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
        video_list.append(database_items)
        # print("database_items",database_items)
    return video_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    
    parser.add_argument('--data_path', metavar='path', required=True,
                        help='the path to the dataset ')   
    parser.add_argument('--save_path', metavar='path', required=True,
                        help='the path to the saved cropped videos')
    parser.add_argument('--csv_output', metavar='path', required=True,
                        help='the path to the generated csv file which contain ("video_name","label")')

    args = parser.parse_args()

    dir_path = args.data_path
    save_path = args.save_path
    csv_file = args.csv_output


    with open(csv_file,"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["video_name","label"])

        for i in os.listdir(dir_path):
            users_dir = os.path.join(dir_path,i)

            for j in os.listdir(users_dir):
                if j[-4:]==".csv":           
                    video_items = load_anno_csv_(os.path.join(users_dir,j))
                    for p in range(len(video_items)):
                        for q in video_items[p]:
                            tmp = q.split("_")
                            if tmp[0]=="Dashboard":
                                for a in range(len(video_items[p][q]["annotations"])):
                                    ss = video_items[p][q]["annotations"][a]["segment"]     
                                    la = video_items[p][q]["annotations"][a]["label"]                         
                                    splited_video_name = "Dashboard"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+"_split_"+str(a)+".MP4"
                                    video_name = "Dashboard"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                                    min_temp = ss[0].split(":")[1]
                                    start = ss[0].split(":")[2]
                                    min_temp1 = ss[1].split(":")[1]
                                    end = ss[1].split(":")[2]
                                    command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format(os.path.join(users_dir,video_name),
                                                            min_temp,start,min_temp1,end,
                                                            os.path.join(save_path,splited_video_name))
                                    # command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format((users_dir),
                                    #                         min_temp,start,min_temp1,end,
                                    #                         os.path.join(save_path,splited_video_name))
                                    os.system(command)
                                    writer.writerow([splited_video_name,la])
                            if tmp[0]=="Rearview":
                                for a in range(len(video_items[p][q]["annotations"])):
                                    ss = video_items[p][q]["annotations"][a]["segment"]     
                                    la = video_items[p][q]["annotations"][a]["label"]                         
                                    splited_video_name = "Rearview"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+"_split_"+str(a)+".MP4"
                                    video_name = "Rear_view"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                                    min_temp = ss[0].split(":")[1]
                                    start = ss[0].split(":")[2]
                                    min_temp1 = ss[1].split(":")[1]
                                    end = ss[1].split(":")[2]
                                    command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format(os.path.join(users_dir,video_name),
                                                            min_temp,start,min_temp1,end,
                                                            os.path.join(save_path,splited_video_name))
                                    os.system(command)
                                    writer.writerow([splited_video_name,la])
                            if tmp[0]=="Rear":
                                for a in range(len(video_items[p][q]["annotations"])):
                                    ss = video_items[p][q]["annotations"][a]["segment"]     
                                    la = video_items[p][q]["annotations"][a]["label"]                         
                                    splited_video_name = "Rearview"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+"_split_"+str(a)+".MP4"
                                    video_name = "Rear_view"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                                    min_temp = ss[0].split(":")[1]
                                    start = ss[0].split(":")[2]
                                    min_temp1 = ss[1].split(":")[1]
                                    end = ss[1].split(":")[2]
                                    command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format(os.path.join(users_dir,video_name),
                                                            min_temp,start,min_temp1,end,
                                                            os.path.join(save_path,splited_video_name))
                                    # command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format((users_dir),
                                    #                             min_temp,start,min_temp1,end,
                                    #                             os.path.join(save_path,splited_video_name))
                                    os.system(command)
                                    writer.writerow([splited_video_name,la])
                            if tmp[0]=="Right":
                                for a in range(len(video_items[p][q]["annotations"])):
                                    ss = video_items[p][q]["annotations"][a]["segment"]     
                                    la = video_items[p][q]["annotations"][a]["label"]                         
                                    splited_video_name = "Right_side_window"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+"_split_"+str(a)+".MP4"
                                    video_name = "Right_side_window"+"_user_id_"+tmp[-2]+"_NoAudio_"+tmp[-1]+".MP4"
                                    min_temp = ss[0].split(":")[1]
                                    start = ss[0].split(":")[2]
                                    min_temp1 = ss[1].split(":")[1]
                                    end = ss[1].split(":")[2]
                                    command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format(os.path.join(users_dir,video_name),
                                                            min_temp,start,min_temp1,end,
                                                            os.path.join(save_path,splited_video_name))
                                    # command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -vcodec copy -acodec copy {}'.format((users_dir),
                                    #                         min_temp,start,min_temp1,end,
                                    #                         os.path.join(save_path,splited_video_name))
                                    os.system(command)
                                    writer.writerow([splited_video_name,la])

    print("Done!")
                                
