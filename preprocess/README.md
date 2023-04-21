# AIcity_Track3

## preprocess

cd preprocess

## train

1.generate json file of labels

run :

python get_jsons_only_rearView.py  --data_path 'the path to the A1 dataset' --lable_path 'the path to the annotation files of A1 ' --json_output 'the path to the generated json file' --data2_path 'the path to the A2 dataset '


2.crop the human body of the input videos

run:

python yolov5/driver_tracking.py --vid_path 'specify videos path based on the workspace'  --out_file 'specify the path of output videos based on the workspace'


3.trip videos 

run:

python get_splited_videos.py --data_path 'the path to the dataset' --save_path ' the path to save the cropped videos' --csv_output 'the path to the generated csv file which contain ("video_name","label") '


## test
1.generate json file of B dataset

run:

python get_jsons_for_dataB.py  --data_path 'the path to the B dataset' --lable_path 'the path to the annotation files of B ' --json_output 'the path to the generated json file' 

2.crop the human body of the input videos

run:

python yolov5/driver_tracking.py --vid_path 'specify videos path based on the workspace'  --out_file 'specify the path of output videos based on the workspace'
