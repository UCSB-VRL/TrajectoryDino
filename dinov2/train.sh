#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
config_file=/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2/train/config.yaml
output_dir=/mnt/mind_ssd2/bowen/projects/haystac_dinov2/dinov2_outputs
export PYTHONPATH=/data/home/bowen/projects/dinov2_trajectory/dinov2
###### no distributed training
# python dinov2/train/train.py --config-file $config_file \
# --output_dir $output_dir 


###### distributed training
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
dinov2/train/train.py --config-file=$config_file --output-dir=$output_dir \
