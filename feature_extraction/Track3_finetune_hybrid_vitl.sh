# Set the path to save checkpoints
OUTPUT_DIR='/mnt/home/Projects/aicity_track3/feature_extraction/workdir/track3_videomae_pretrain_hybrid_l_pt_800e_k700/A1_Mix_lr_1e-3_epoch_35'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/AIcity/cropped_annotations_A1'
# path to pretrain model
#MODEL_PATH='/mnt/home/VideoMAE/workdir/track3_videomae_pretrain_hybrid_l_224_f16x2_mask_0.9_e400/checkpoint-239.pth'
MODEL_PATH='./pretrain/vit_l_hybrid_pt_800e_k700_ft.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Track3 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 2e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 35 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --enable_deepspeed