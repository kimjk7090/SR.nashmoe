angle_version = 'le90'
data_root = './'
dataset_type = 'PEdataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=100, type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='re_resnet50_c8_batch256-25b16846.pth',
            type='Pretrained'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ReResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='ReFPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    angle_version='le90',
                    edge_swap=True,
                    norm_factor=2,
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                        0.1,
                    ),
                    type='DeltaXYWHTHBBoxCoder',
                    use_box_type=True),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(
                    beta=1.0, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=1,
                predict_box_type='rbox',
                reg_class_agnostic=True,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                roi_feat_size=7,
                type='mmdet.Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    angle_version='le90',
                    edge_swap=True,
                    norm_factor=None,
                    proj_xy=True,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                        0.05,
                    ],
                    type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(
                    beta=1.0, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=1,
                predict_box_type='rbox',
                reg_class_agnostic=False,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                roi_feat_size=7,
                type='mmdet.Shared2FCBBoxHead'),
        ],
        bbox_roi_extractor=[
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='mmdet.SingleRoIExtractor'),
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    clockwise=True,
                    num_orientations=8,
                    num_samples=2,
                    out_size=7,
                    type='RiRoIAlignRotated'),
                type='RotatedSingleRoIExtractor'),
        ],
        num_stages=2,
        stage_loss_weights=[
            1,
            1,
        ],
        type='mmdet.CascadeRoIHead'),
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
            type='mmdet.AnchorGenerator',
            use_box_type=True),
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
            type='DeltaXYWHHBBoxCoder',
            use_box_type=True),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        type='mmdet.RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            nms_pre=2000,
            score_thr=0.05),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='mmdet.RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='mmdet.RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='mmdet.RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='mmdet.CascadeRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=25, norm_type=2),
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
pretrained = 'work_dirs/pretrain/re_resnet50_c8_batch256-25b16846.pth'
randomness = dict(seed=3407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val/',
        data_prefix=dict(img_path='val_images/'),
        data_root='./',
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='PEdataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=10, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='train_test/',
        data_prefix=dict(img_path='train_test_images/'),
        data_root='./',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='PEdataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val/',
        data_prefix=dict(img_path='val_images/'),
        data_root='./',
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='PEdataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './run/train/pd_redet_240314/'
