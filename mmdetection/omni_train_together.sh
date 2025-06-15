#!/bin/bash

# # faster-rcnn
# # faster-rcnn_r50_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni.py \
# --launcher pytorch > log3/log_faster-rcnn_r50_fpn_100_omni.log

# # faster-rcnn_r101_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r101_fpn_100_omni.py \
# --launcher pytorch > log3/log_faster-rcnn_r101_fpn_100_omni.log

# faster-rcnn_r101_fpn_100_omni_RoIPool
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r101_fpn_100_omni_RoIPool.py \
--launcher pytorch > log3/log_faster-rcnn_r101_fpn_100_omni_RoIPool.log

# # faster-rcnn_r50_fpn_100_omni_RoIPool
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni_RoIPool.py \
# --launcher pytorch > log3/log_faster-rcnn_r50_fpn_100_omni_RoIPool.log

# # mask_rcnn
# # mask-rcnn_r50_fpn_100_omni_RoIPool
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni_RoIPool.py \
# --launcher pytorch > log3/log_mask-rcnn_r50_fpn_100_omni_RoIPool.log

# # mask-rcnn_r50_fpn_100_omni_without_FPN
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni_without_FPN.py \
# --launcher pytorch > log3/log_mask-rcnn_r50_fpn_100_omni_without_FPN.log

# # mask-rcnn_r50_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni.py \
# --launcher pytorch > log3/log_mask-rcnn_r50_fpn_100_omni.log

# # mask-rcnn_r101_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r101_fpn_100_omni.py \
# --launcher pytorch > log3/log_mask-rcnn_r101_fpn_100_omni.log

# # deformable_detr
# # deformable_detr_r50_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/deformable_detr/deformable_detr_r50_100_omni.py \
# --launcher pytorch > log3/log_deformable_detr_r50_100_omni.log

# # deformable_detr_r101_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/deformable_detr/deformable_detr_r101_100_omni.py \
# --launcher pytorch > log3/log_deformable_detr_r101_100_omni.log

# # efficientdet
# # efficientdet_effb3_bifpn_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb3_bifpn_omni.py \
# --launcher pytorch > log3/log_efficientdet_effb3_bifpn_omni.log

# # efficientdet_effb3_omni_without_bifpn
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb3_omni_without_bifpn.py \
# --launcher pytorch > log3/log_efficientdet_effb3_omni_without_bifpn.log

# # efficientdet_effb0_bifpn_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb0_bifpn_omni.py \
# --launcher pytorch > log3/log_efficientdet_effb0_bifpn_omni.log

# # detr
# # detr_r50_300_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r50_300_omni.py \
# --launcher pytorch > log3/log_detr_r50_300_omni.log

# # detr_r101_300_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r101_300_omni.py \
# --launcher pytorch > log3/log_detr_r101_600_omni.log

# # detr_r50_300_omni_without_positional_encoding
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r50_300_omni_without_positional_encoding.py \
# --launcher pytorch > log3/log_detr_r50_300_omni_without_positional_encoding.log

