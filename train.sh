# CUDA_VISIBLE_DEVICES=7 python dinov2/run/train/train.py \
#     --nodes 1 \
#     --config-file dinov2/configs/train/vitl16_short.yaml \
#     --output-dir runs/run1 \
#     train.dataset_path=ImageNet:split=TRAIN:root=datasets/imagenette2:extra=datasets/imagenette2_extra


# CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 dinov2/train/train.py \
#     --config-file=dinov2/configs/train/vitl16_short.yaml \
#     --output-dir=runs/run1 \ 


CUDA_VISIBLE_DEVICES=4,5,6,7 python dinov2/train/train.py \
    --gpus-per-node=4 \
    --output-dir=runs/run1 \ 