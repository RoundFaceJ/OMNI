MAX_EPOCH = 50
NUM_WORKERS = 32
auto_scale_lr = dict(base_batch_size=32, enable=False) #
backend_args = None
data_root = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/data/omni' # 
dataset_type = 'OMNIDataset' #
default_hooks = dict( #
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每个 epoch 保存一次检查点
        save_best='coco/bbox_mAP_50',  # 基于 bbox_mAP 保存最佳模型
        rule='greater',
        max_keep_ckpts=1,  # 最多保存的模型数量
    ),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
randomness = dict(seed=0)
env_cfg = dict(
    deterministic = True,
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = "/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth" #
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=10,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, type='RoIPool'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
# optim_wrapper = dict( #
#     optimizer=dict(type='AdamW',lr=0.0001,weight_decay=0.05,betas=(0.9, 0.999)),
#     # optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
#     type='OptimWrapper')
# param_scheduler = [ #
#     dict(
#         begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
#     dict(
#         begin=0,
#         by_epoch=True,
#         end=MAX_EPOCH,
#         gamma=0.1,
#         milestones=[
#             16,
#             22,
#         ],
#         type='MultiStepLR'),
# ]

# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.05),
#     type='OptimWrapper'
# )

# param_scheduler = [
#     # Warm-up 线性学习率策略，按 epoch 调整
#     dict(
#         type='LinearLR',
#         begin=0,      # 从第 0 个 epoch 开始
#         end=1,        # 第 1 个 epoch 结束热身
#         start_factor=0.001,  # 学习率起始为 0.001 倍
#         by_epoch=True
#     ),
#     # CosineAnnealingLR 学习率调度策略
#     dict(
#         type='CosineAnnealingLR',
#         begin=1,     # 从第 1 个 epoch 开始余弦退火
#         T_max=MAX_EPOCH,    # 余弦退火周期为 10 个 epoch
#         eta_min=0,   # 最小学习率为 0
#         by_epoch=True
#     )
# ]

optim_wrapper = dict( #
    optimizer=dict(type='AdamW',lr=0.0001,weight_decay=0.05,betas=(0.9, 0.999)),
    # optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [ #
    dict(
        begin=0, by_epoch=False, end=300, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=MAX_EPOCH,
        gamma=0.1,
        milestones=[15, 30, 40],
        type='MultiStepLR'),
]

resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict( #
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_test.json', #
        backend_args=None,
        data_prefix=dict(img='test/'), #
        data_root=data_root, #  
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict( #
    ann_file=data_root+'/annotations/instances_test.json', #
    backend_args=None,
    format_only=False,
    metric='bbox',
    classwise=True,
    type='OMNIMetric')
test_pipeline = [ #
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=MAX_EPOCH, type='EpochBasedTrainLoop', val_interval=1) # 
train_dataloader = dict( #
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train.json', #
        backend_args=None,
        data_prefix=dict(img='train/'), #
        data_root=data_root, #
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type=dataset_type),
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [ #
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop') #
val_dataloader = dict( #
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_val.json', #
        backend_args=None,
        data_prefix=dict(img='val/'), #
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict( #
    ann_file=data_root+'/annotations/instances_val.json', #
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='OMNIMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/faster-rcnn/faster-rcnn_r50_fpn_100_omni_RoIPool' #
