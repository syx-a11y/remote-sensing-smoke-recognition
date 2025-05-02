# model settings

model_cfg = dict(
    backbone=dict(
        type='MFResNet',
        num_stages=4,
        depth=50,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,  # This will be set based on the actual output channels of CFResNet34.
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataloader pipeline
img_norm_cfg = dict(
    mean=[0.47844062, 0.48068325, 0.47762897], std=[0.21663591, 0.21464501, 0.21836974], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size=32,
    num_workers=4,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=('backbone',),
        epoches=100,
    ),
    val=dict(
        ckpt='',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options=dict(
            topk=(1, 5),
            thrs=None,
            average_mode='none'
        )
    ),
    test=dict(
        ckpt='logs/MFResNet/2025-02-03-11-13-44/Train_Epoch097-Loss0.095.pth',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion']
    )
)

# batch 32
# lr = 0.1 *32 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 32/256,
    momentum=0.9,
    weight_decay=1e-4)
# learning
lr_config = dict(
    type='StepLrUpdater',
    step=[40, 80, 120]
)