_base_ = './deformable_detr_r50_100_omni.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

auto_scale_lr = dict(base_batch_size=16, enable=False) #
test_dataloader = dict(batch_size=4)
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=4)

load_from = None
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/deformable_detr/deformable_detr_r101_100_omni' #
