_base_ = './mask-rcnn_r50_fpn_100_omni.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/mask-rcnn/mask-rcnn_r101_fpn_100_omni'
load_from = "/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth"