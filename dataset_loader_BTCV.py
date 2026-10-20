import json

from monai.data import (
    CacheDataset,
)
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandShiftIntensityd,
    RandSpatialCropd
)


class Augment:
    def __init__(self):
        self.aug = Compose([
            RandSpatialCropd(
                keys=["image", "scrib"],
                roi_size=[96, 96, 96],
                random_size=False,
            ),
            RandFlipd(
                keys=["image", "scrib"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "scrib"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "scrib"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "scrib"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            )
        ])

    def __call__(self, sample):
        return self.aug(sample)


def get_dataset_BTCV(data_path=None):
    with open(f"{data_path}/dataset.json") as fp:
        data_dic = json.load(fp)
    train_data_list = []
    val_data_list = []
    for d_name in data_dic["train"]:
        train_data_list.append({
            "image": f"{data_path}/images/{d_name['image']}",
            "scrib": f"{data_path}/scrib/{d_name['scrib']}"
        })

    for d_name in data_dic["test"]:
        val_data_list.append({
            "image": f"{data_path}/images/{d_name['image']}",
            "label": f"{data_path}/labels/{d_name['label']}"
        })

    train_transform = [
        LoadImaged(keys=["image", "scrib"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "scrib"], source_key="image"),
        Orientationd(keys=["image", "scrib"], axcodes="RAS"),
        Spacingd(
            keys=["image", "scrib"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "scrib"], device="cuda", track_meta=False),
        Augment()
    ]
    train_ds = CacheDataset(
        data=train_data_list,
        transform=Compose(train_transform),
        cache_num=24,
        cache_rate=1.0,
        num_workers=16,
    )
    val_transform = [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device="cuda", track_meta=False)
    ]
    val_ds = CacheDataset(
        data=val_data_list,
        transform=Compose(val_transform),
        cache_num=6,
        cache_rate=1.0,
        num_workers=16
    )
    return train_ds, val_ds
