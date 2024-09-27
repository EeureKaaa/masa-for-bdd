#!/bin/bash

# Set the CUDA device(s) to be visible to the script
export CUDA_VISIBLE_DEVICES=0,1  # Change this to the appropriate GPU(s) you want to use

# Define arguments
VIDEO_DIR="/home/wxh/data_wxh/hhz/bdd_seg/videos/val"
OUTPUT_DIR="/home/wxh/data_wxh/hhz/bdd_seg_masa/bonding_box/clip/val"
CONFIG_FILE="/home/wxh/data_wxh/hhz/masa/configs/masa-gdino/masa_gdino_swinb_inference.py"
CHECKPOINT_FILE="/home/wxh/data_wxh/hhz/masa/saved_models/masa_models/gdino_masa.pth"
SCORE_THR=0.15
TARGET_WIDTH=128
IMAGE_WIDTH=128
NUM_SUB_PROCESS_PER_GPU=2
MAX_OBJECT_NUM=64 #todo ?
EP_LEN=4
BATCH_SIZE=4
VIDEO_EXT=".mp4"
PHASE=""
USE_MASA=true  # Change to true if you want to enable this option

# Run the Python script with the specified arguments
python test.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_file "$CONFIG_FILE" \
    --checkpoint_file "$CHECKPOINT_FILE" \
    --score_thr "$SCORE_THR" \
    --target_width "$TARGET_WIDTH" \
    --image_width "$IMAGE_WIDTH" \
    --num_sub_process_per_gpu "$NUM_SUB_PROCESS_PER_GPU" \
    --max_object_num "$MAX_OBJECT_NUM" \
    --ep_len "$EP_LEN" \
    --batch_size "$BATCH_SIZE" \
    --video_ext "$VIDEO_EXT" \
    --phase "$PHASE" \
    --use_masa "$USE_MASA"
