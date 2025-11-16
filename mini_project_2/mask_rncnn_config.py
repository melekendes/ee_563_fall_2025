# mask_rncnn_config.py (TEMİZ KOD)

_base_ = './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

dataset_type = 'CocoDataset'
data_root = 'balloon/'
classes = ('balloon',)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotation_coco.json',
        img_prefix=data_root + 'train/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        classes=classes))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

evaluation = dict(metric=['bbox', 'segm'])
work_dir = './work_dirs/mask_rcnn_balloon'