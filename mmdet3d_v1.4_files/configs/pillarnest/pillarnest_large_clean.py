_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/models/centerpoint_pillar02_second_secfpn_nus_pillarnest.py',
    '../_base_/schedules/cyclic-20e.py', 
    '../_base_/default_runtime.py'
]


# 1. REMOVED ADVERSARIAL HOOKS
# custom_hooks = [...] 

voxel_size = [0.15, 0.15, 8]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

#your custom imports for the PillarNest modules
custom_imports = dict(
    imports=[
        'mmdet3d.models.backbones.pillarnest_convnext',
        'mmdet3d.models.voxel_encoders.pillarnest_pillar_encoder',
        'mmdet3d.models.dense_heads.pillarnest_center_head',
        'mmdet3d.models.task_modules.coders.pillarnest_bbox_coder',
        # 'mmdet3d.engine.optimizers.my_optimizer' # Removed custom optimizer import
    ],
    allow_failed_imports=False)
work_dir = ''

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        _delete_=True,
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20, # NuScenes Standard
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000))),
    
    pts_voxel_encoder=dict(
        _delete_=True,
        type='PillarNestHeightFeatureNet',
        in_channels=5, # NuScenes Standard (x,y,z,i,t)
        feat_channels=[96],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='maxavg',
        encoder_layer='PFNLayer',
        legacy=False),
    
    pts_middle_encoder=dict(
        _delete_=True,
        type='PointPillarsScatter',
        in_channels=96,
        output_shape=(720, 720)),
    
    pts_backbone=dict(
        _delete_=True,
        type='PillarNestConvNeXt',
        arch='large',
        in_channels=96,
        stem_patch_size=4,
        norm_cfg=dict(type='LN2d', eps=1e-6),
        act_cfg=dict(type='GELU'),
        linear_pw_conv=True,
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[2, 3, 4],
        frozen_stages=0,
        gap_before_final_norm=False,
        first_downsample=1,
        large_arch=None,
        # init_cfg=dict(
        # type="Pretrained",
        # # Use the TINY checkpoint, not large
        # checkpoint="/home/yerdana/links/projects/rrg-instructor/yerdana/checkpoint/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth", 
        # prefix="backbone.",)
    ),
    
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[384, 384, 384],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    pts_bbox_head=dict(
        _delete_=True,
        type='CenterHead',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2),
            #iou=(1, 2)),
            ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='PillarNestBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4, # Corrected to 4
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        #iou_score=dict(type='BboxOverlaps3D', coordinate='lidar'),
        #loss_iou_score=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=1.0),
        #iou_score_weight=1.0,
        norm_bbox=True),
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            grid_size=[720, 720, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4, # Corrected to 4
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    
    test_cfg=dict(
        _delete_=True,
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4, # Corrected to 4
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            iou_score_beta=0.5)))

# Dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    # 2. DISABLED OBJECT SAMPLE (Matches PillarNest Source Config)
    # dict(
    #     type='ObjectSample',
    #     db_sampler=dict( ... )
    # ),
    # 3. RESTORED STANDARD AUGMENTATIONS
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],   # Restored Source Rotation
        scale_ratio_range=[0.95, 1.05], # Restored Source Scale
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',            # Restored Flipping
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=class_names),
    dict(type='PointShuffle'),          # Restored Shuffle
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

train_dataloader = dict(
        dataset=dict(
            pipeline=train_pipeline, 
            metainfo=dict(classes=class_names), 
            ann_file='nuscenes_infos_train.pkl' # Ensure this is train pkl
        ),
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=True))

val_evaluator = dict(
    _delete_=True,
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# 4. RESTORED STANDARD OPTIMIZER (Matches Source)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# 5. REMOVED MANUAL SCHEDULER 
# (Letting _base_/schedules/cyclic-20e.py handle it)
# param_scheduler = ...


