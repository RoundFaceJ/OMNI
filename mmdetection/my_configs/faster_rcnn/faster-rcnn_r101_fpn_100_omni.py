_base_ = './faster-rcnn_r50_fpn_100_omni.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
load_from = "/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth"
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r101_fpn_100_omni'