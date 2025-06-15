# 在./detr_r50_300_omni.py中修改MAX_EPOCH = 600
_base_ = './detr_r50_300_omni.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

load_from = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/detr/detr-r101-2c7b67e5.pth' # https://github.com/facebookresearch/detr
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/detr/detr_r101_900_omni'