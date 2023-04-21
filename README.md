# AIcity_Track3



## Trainning
The workflow for training action classification model is as follow:

1.Dataset preparation
- Detect driver spatial location in the video, then crop each video based on the driver bounding box.
- Trimming Videos the input videos should be a trimmed videos i.e., contains only one action in each video.
- Prepare csv Files for the training and validation sets.

2.Featrue extraction
- Download open source pre-training weights.
- Finetune training using A1 dataset.
- Extracting A2 video features using trained weights.

3.action_detection_code
- Using the features to train the task of temporal action localization.
- Generating the action location csv files with start and end time.


4.Time correction
- get txt file of one TAL model result.
- get final merged txt file of multi txt.


## Test on B dataset
The workflow for testing action classification model is as follow:

1.Dataset preparation
- crop the human body of the input videos

>> run:
>>```shell  
>>python yolov5/driver_tracking.py --vid_path 'specify videos path based on the workspace'  --out_file 'specify the path of output videos based on the workspace'
>>``` 

- generate json file of B dataset

>> run:
>>```shell  
>>python get_jsons_for_dataB.py  --data_path 'the path to the B dataset' --lable_path 'the path to the annotation files of B ' --json_output 'the path to the generated json file' 
>>``` 


2.Featrue extraction
- Download weights.
- Extracting video features of B dataset using trained weights.
>> Firstly, to extract video features using ViT-H on rear view and dash view of official videos, you can run:

>>```shell
 python inference_video_feature_vithK400.py --ckpt_pth ./weights/k400_vith_rearview.pt --video_dir XXX --output_dir XXX --select_view Rear --device cuda:0  
 python inference_video_feature_vithK400.py --ckpt_pth ./weights/K400_vith_dashboard.pt --video_dir XXX --output_dir XXX --select_view Dash --device cuda:0  
>>```  

>> Secondly, to extract video features using ViT-L on rear view and dash view of official videos, you can run:

>>```shell 
 >>python inference_video_feature_vitl.py  --model_path ./weights/hybrid_k700_vitl_rearview.pt --video_dir XXX --save_dir XXX --view Rear --device cuda:0  
 >>python inference_video_feature_vitl.py  --model_path ./weights/hybrid_k700_vitl_dashboard.pt --video_dir XXX --save_dir XXX --view Dash --device cuda:0  
 >>python inference_video_feature_vitl.py  --model_path ./weights/ego_verb_vitl_rearview.pt --video_dir XXX --save_dir XXX --view Rear --device cuda:0  
 >>python inference_video_feature_vitl.py  --model_path ./weights/ego_verb_vitl_dashboard.pt --video_dir XXX --save_dir XXX --view Dash --device cuda:0  
>>``` 


3.action_detection_code
- Modify the relevant config file(./configs/aicity_action_xxx.yaml), change the path of "feat_folder" and "json_file".


- Generating the action location csv files with start and end time.   
>> cd ./MA-Actionformer  
>>```shell      
>>python ./eval.py ./configs/aicity_action_k400.yaml ./ckpt/aicity_action_vmae_vitHK400_3modelAIcityA1_1280_crop_rear_A1-train_A2-infe        
>>python ./eval.py ./configs/aicity_action_ego.yaml ./ckpt/aicity_action_ego4d_verb_vitl_track3_crop_pred_rear_A1-train_A2-infe     
>>python ./eval.py ./configs/aicity_action_hybird.yaml ./ckpt/aicity_action_hybrid_k700_vitl_track3_crop_pred_e35_A1-train_A2-infe   
>>```  

>>cd ./tridet       
>>```shell      
>>python ./eval.py ./configs/aicity_action.yaml ./ckpt/aicity_videomae_vitHK400_3modelAIcityA1_1280+16_personOnly_A1-train_A2-infe_tridet     
>>```   


4.Time correction
- get txt file of one TAL model result.

>> run: 
>>```shell      
>>python get_final_txt_from_csv.py  --csv_path 'the path to the csv file got in action_detection_code' --out_file 'the file name of the generated txt file'
>>```  

- get final merged txt file of multi txt.

>> run:
>>```shell      
>>python merge_txt.py --txt_path_list 'the paths to the txt file ' --out_file 'the file name of the generated txt file'   
>>```  

