#!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5
# config_file=config_files.yaml
output_dir=/mnt/mind_ssd2/bowen/projects/haystac_dinov2/dinov2_outputs

# python train.py --config_file $config_file \
# --output_dir $output_dir --gpus-per-node 2

config_file=/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2/train/config.yaml
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/data/home/bowen/projects/dinov2_trajectory/dinov2
python -m torch.distributed.launch --nproc_per_node=2 \
/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2/train/train.py --config-file=$config_file --output-dir=$output_dir
