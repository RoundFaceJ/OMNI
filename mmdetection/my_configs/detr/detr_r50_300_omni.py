MAX_EPOCH = 300
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
launcher = 'pytorch'
load_from = "/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/checkpoints/detr/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        embed_dims=256,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            bg_cls_weight=0.1,
            class_weight=1.0,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=10,
        type='DETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6,
        return_intermediate=True),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1,
                num_fcs=2),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            2048,
        ],
        kernel_size=1,
        norm_cfg=None,
        num_outs=1,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=100,
    positional_encoding=dict(normalize=True, num_feats=128),
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DETR')
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
        begin=0, by_epoch=False, end=5000, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=MAX_EPOCH,
        gamma=0.1,
        milestones=[300, 600, 800],
        type='MultiStepLR'),
]

resume = None
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
    classwise=True,
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
work_dir = '/mnt/nvme0n1/xpj_files/project/OMNI/mmdetection/work_dirs3/detr/detr_r50_900_omni' #