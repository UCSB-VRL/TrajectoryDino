export CUDA_VISIBLE_DEVICES=4,5
config_file=/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2/train/config.yaml
output_dir=/mnt/mind_ssd2/bowen/projects/haystac_dinov2/dinov2_outputs
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file $config_file \
    --output-dir $output_dir \
