MAX_EPOCH = 50
NUM_WORKERS = 32
auto_scale_lr = dict(base_batch_size=16, enable=False) #
backend_args = None
batch_augments = [
    dict(size=(
        896,
        896,
    ), type='BatchFixedSizePad'),
]
load_from = "/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco_20230223_122457-e6f7a833.pth"
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.EfficientDet.efficientdet',
    ])
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
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        arch='b3',
        conv_cfg=dict(type='Conv2dSamePadding'),
        drop_path_rate=0.3,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        norm_eval=False,
        out_indices=(
            3,
            4,
            5,
        ),
        type='EfficientNet'),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=4,
            ratios=[
                1.0,
                0.5,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
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
        feat_channels=160,
        in_channels=160,
        loss_bbox=dict(beta=0.1, loss_weight=50, type='HuberLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=1.5,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_classes=10,
        num_ins=5,
        stacked_convs=4,
        type='EfficientDetSepBNHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(size=(
                896,
                896,
            ), type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            48,
            136,
            384,
        ],
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_stages=6,
        out_channels=160,
        start_level=0,
        type='BiFPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(
            iou_threshold=0.3,
            method='gaussian',
            min_score=0.001,
            sigma=0.5,
            type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='EfficientDet')
norm_cfg = dict(eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN')
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
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_test.json', #
        backend_args=None,
        data_prefix=dict(img='test/'), #
        data_root=data_root, #  
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                896,
                896,
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
    dict(keep_ratio=False, scale=(
        896,
        896,
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
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_train.json', #
        backend_args=None,
        data_prefix=dict(img='train/'), #
        data_root=data_root, #
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                896,
                896,
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
    dict(keep_ratio=False, scale=(
        896,
        896,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop') #
val_dataloader = dict( #
    batch_size=4,
    dataset=dict(
        ann_file='annotations/instances_val.json', #
        backend_args=None,
        data_prefix=dict(img='val/'), #
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                896,
                896,
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
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/efficientdet/efficientdet_effb3_bifpn_omni' #