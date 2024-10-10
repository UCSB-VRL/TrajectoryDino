#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
config_file=vit16_short_traj.yaml
# dinov2/configs/train/vitl16_short.yaml
output_dir=/mnt/mind_ssd2/bowen/projects/haystac_dinov2/dinov2_outputs
python train.py 
