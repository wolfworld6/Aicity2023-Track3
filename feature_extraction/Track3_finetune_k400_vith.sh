OUTPUT_DIR='./workdir/list_K400_2e3bs1-PersonOnly_Rearview_s448_f16'
# path to pretrain model
MODEL_PATH='./pretrain/k400_videomae_vitH_pre_finetune.pth'
DATA_PATH='/AIcity/cropped_annotations_A1'
# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_huge_patch16_448 \
    --data_set Track3 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 2 \
    --input_size 448 \
    --short_side_size 448 \
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 2e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --drop_path 0.2  \
    --fc_drop_rate 0.5 \
    --test_num_segment 1 \
    --test_num_crop 3 \
    --dist_eval \
    --use_checkpoint \
    --enable_deepspeed 
