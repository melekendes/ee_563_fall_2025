# 1. TEMEL KONFİGÜRASYON
_base_ = './configs/ssd/ssd300_coco.py'

# 2. MODEL (Sınıf sayısı ve çapa ayarları)
model = dict(
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            basesize_ratio_range=(0.2, 0.9),
            ratios=[[2], [2, 0.5], [2, 0.5], [2, 0.5], [2], [2]]
        )
    )
)

# 3. VERİ SETİ (v2.x Yapısı)
dataset_type = 'CocoDataset'
data_root = 'balloon/'
classes = ('balloon',)

# 4. VERİ YÜKLEYİCİLER (Dataloaders)
# 'RepeatDataset' sarmalayıcısı 'times' hatasını önlemek için korundu
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5, 
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train/annotation_coco.json',
            img_prefix=data_root + 'train/',
            classes=classes,
            # 'pipeline' temel konfigürasyondan miras alınır
        )),
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


# 5. YÜKLEME (İndirdiğimiz yerel dosyanın adı)
load_from = 'ssd300_coco_20210803_015428-d231a06e.pth'

# 6. Değerlendirme (Sadece 'bbox' mAP hesaplayacak)
evaluation = dict(metric='bbox')

# 7. Çalışma Dizini
work_dir = './work_dirs/ssd300_balloon'