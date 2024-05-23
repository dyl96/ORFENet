dataset_type = 'AITODv2Dataset'
data_root = '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='AITODv2Dataset',
        ann_file=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/annotations_v2/aitodv2_trainval.json',
        img_prefix=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='AITODv2Dataset',
        ann_file=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/annotations_v2/aitodv2_test.json',
        img_prefix=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='AITODv2Dataset',
        ann_file=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/annotations_v2/aitodv2_test.json',
        img_prefix=
        '/home/ldy/mmdet_ldy/mmdetection-master/data/AI-TOD/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=12, metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10000,
    warmup_ratio=0.3333333333333333,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=36)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='FCOS',
    add_or=True,
    add_hrfe=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        norm_cfg=None,
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
work_dir = './work_dirs/aitodv2_fcos_r50_p2_hrfe_or_3x_b2'
auto_resume = False
gpu_ids = [1]
