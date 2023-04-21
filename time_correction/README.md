# AIcity_Track3


#1.get txt file of one model result

run: 

python get_final_txt_from_csv.py  --csv_path 'the path to the csv file got in action_detection_code' --out_file 'the file name of the generated txt file'

for example:

python merge_txt.py --txt_path_list "/mnt/home/dataset/process_track3/txt_files/ego_4_14.txt" --txt_path_list "/mnt/home/dataset/process_track3/txt_files/hyb_4_14.txt" --txt_path_list "/mnt/home/dataset/process_track3/txt_files/hug_4_14.txt"  --out_file "/mnt/home/dataset/process_track3/txt_files/ehh_4_19.txt"

#2.get merged txt file of multi txt got in step 1

run:

python merge_txt.py --txt_path_list 'the paths to the txt file ' --out_file 'the file name of the generated txt file'

