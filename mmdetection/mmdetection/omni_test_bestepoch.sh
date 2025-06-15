#!/bin/bash

# # faster-rcnn
# # faster-rcnn_r50_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r50_fpn_100_omni/best_coco_bbox_mAP_50_epoch_16.pth \
# --launcher pytorch > log3/test/best_epoch/log_faster-rcnn_r50_fpn_100_omni.log

# # faster-rcnn_r101_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r101_fpn_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r101_fpn_100_omni/best_coco_bbox_mAP_50_epoch_16.pth \
# --launcher pytorch > log3/test/best_epoch/log_faster-rcnn_r101_fpn_100_omni.log

# # faster-rcnn_r50_fpn_100_omni_RoIPool
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni_RoIPool.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r50_fpn_100_omni_RoIPool/best_coco_bbox_mAP_50_epoch_15.pth \
# --launcher pytorch > log3/test/best_epoch/log_faster-rcnn_r50_fpn_100_omni_RoIPool.log

# faster-rcnn_r101_fpn_100_omni_RoIPool
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/faster_rcnn/faster-rcnn_r101_fpn_100_omni_RoIPool.py \
/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r101_fpn_100_omni_RoIPool/best_coco_bbox_mAP_50_epoch_16.pth \
--launcher pytorch > log3/test/best_epoch/log_faster-rcnn_r101_fpn_100_omni_RoIPool.log

# # mask_rcnn
# # mask-rcnn_r50_fpn_100_omni_RoIPool
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni_RoIPool.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/mask-rcnn/mask-rcnn_r50_fpn_100_omni_RoIPool/best_coco_bbox_mAP_50_epoch_11.pth \
# --launcher pytorch > log3/test/best_epoch/log_mask-rcnn_r50_fpn_100_omni_RoIPool.log

# # mask-rcnn_r50_fpn_100_omni_without_FPN
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni_without_FPN.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/mask-rcnn/mask-rcnn_r50_fpn_100_omni_without_FPN/best_coco_bbox_mAP_50_epoch_16.pth \
# --launcher pytorch > log3/test/best_epoch/log_mask-rcnn_r50_fpn_100_omni_without_FPN.log

# # mask-rcnn_r50_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r50_fpn_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/mask-rcnn/mask-rcnn_r50_fpn_100_omni/best_coco_bbox_mAP_50_epoch_11.pth \
# --launcher pytorch > log3/test/best_epoch/log_mask-rcnn_r50_fpn_100_omni.log

# # mask-rcnn_r101_fpn_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/mask_rcnn/mask-rcnn_r101_fpn_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/mask-rcnn/mask-rcnn_r101_fpn_100_omni/best_coco_bbox_mAP_50_epoch_16.pth \
# --launcher pytorch > log3/test/best_epoch/log_mask-rcnn_r101_fpn_100_omni.log

# # efficientdet
# # efficientdet_effb0_bifpn_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb0_bifpn_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/efficientdet/efficientdet_effb0_bifpn_omni/best_coco_bbox_mAP_50_epoch_137.pth \
# --launcher pytorch > log3/test/best_epoch/log_efficientdet_effb0_bifpn_omni.log

# # efficientdet_effb3_bifpn_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb3_bifpn_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/efficientdet/efficientdet_effb3_bifpn_omni/best_coco_bbox_mAP_50_epoch_17.pth \
# --launcher pytorch > log3/test/best_epoch/log_efficientdet_effb3_bifpn_omni.log

# # efficientdet_effb3_omni_without_bifpn
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/efficientdet/efficientdet_effb3_omni_without_bifpn.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/efficientdet/efficientdet_effb3_omni_without_bifpn/best_coco_bbox_mAP_50_epoch_19.pth \
# --launcher pytorch > log3/test/best_epoch/log_efficientdet_effb3_omni_without_bifpn.log

# # deformable_detr
# # deformable_detr_r50_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/deformable_detr/deformable_detr_r50_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/deformable_detr/deformable_detr_r50_100_omni/best_coco_bbox_mAP_50_epoch_19.pth \
# --launcher pytorch > log3/test/best_epoch/log_deformable_detr_r50_100_omni.log

# # deformable_detr_r101_100_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/deformable_detr/deformable_detr_r101_100_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/deformable_detr/deformable_detr_r101_100_omni/best_coco_bbox_mAP_50_epoch_18.pth \
# --launcher pytorch > log3/test/best_epoch/log_deformable_detr_r101_100_omni.log

# # detr
# # detr_r50_300_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r50_300_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/detr/detr_r50_900_omni/best_coco_bbox_mAP_50_epoch_117.pth \
# --launcher pytorch > log3/test/best_epoch/log_detr_r50_300_omni.log

# # detr_r101_300_omni
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r101_300_omni.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/detr/detr_r101_900_omni/best_coco_bbox_mAP_50_epoch_305.pth \
# --launcher pytorch > log3/test/best_epoch/log_detr_r101_300_omni.log

# # detr_r50_300_omni_without_positional_encoding
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/test.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/my_configs/detr/detr_r50_300_omni_without_positional_encoding.py \
# /mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/detr/detr_r50_300_omni_without_positional_encoding/best_coco_bbox_mAP_50_epoch_193.pth \
# --launcher pytorch > log3/test/best_epoch/log_detr_r50_300_omni_without_positional_encoding.log