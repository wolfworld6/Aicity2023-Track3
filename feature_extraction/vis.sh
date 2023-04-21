# Set the path to save video
OUTPUT_DIR='/mnt/home/VideoMAE/demo/somethinv2_ViTB_2400/'
# path to video for visualization
VIDEO_PATH='/mnt/home/VideoMAE/test_data/Dashboard_user_id_49989_NoAudio_5.MP4'
# path to pretrain model
#MODEL_PATH='TODO/videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'
MODEL_PATH='/mnt/home/VideoMAE/videoMAE_somethinv2_ViTB_2400_pretrain.pth'
#MODEL_PATH='/mnt/home/VideoMAE/workdir/track3_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-300.pth'


python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}