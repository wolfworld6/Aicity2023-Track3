# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_huge_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/eval_lr_1e-3_epoch_50_droppath_0.2_layer_decay_0.75'
# path to Kinetics set (train.csv/val.csv/test.csv)
OUTPUT_DIR='YOUR_PATH/list_kinetics-400'
# path to pretrain model
MODEL_PATH='YOUR_PATH/k400_videomae_pretrain_huge_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 --nnodes=8  --node_rank=$1 --master_addr=$2 \
    run_class_finetuning.py \
    --model vit_huge_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --layer_decay 0.75 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --drop_path 0.2  \
    --fc_drop_rate 0.5 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --use_checkpoint \
    --enable_deepspeed 
