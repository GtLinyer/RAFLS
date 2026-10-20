import os

from monai.data import (
    CacheDataset,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    RandShiftIntensityd,
    RandSpatialCropd, Spacingd, CropForegroundd
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


class AddChannel:
    def __init__(self, label_type="label"):
        self.label_type = label_type

    def __call__(self, sample):
        sample["image"] = sample["image"].unsqueeze(dim=0)
        sample[self.label_type] = sample[self.label_type].unsqueeze(dim=0)
        return sample


class ShowShape:
    def __init__(self, num="a"):
        self.num = num

    def __call__(self, sample):
        print(self.num, sample["image"].shape)
        return sample


def get_dataset_WORD(data_path=None):
    train_name_list = os.listdir(f"{data_path}/imagesTr")
    val_name_list = os.listdir(f"{data_path}/imagesVal")

    train_data_list = []
    val_data_list = []
    for d_name in train_name_list:
        train_data_list.append({
            "image": f"{data_path}/imagesTr/{d_name}",
            "scrib": f"{data_path}/scribTr/{d_name}"
        })

    for d_name in val_name_list:
        val_data_list.append({
            "image": f"{data_path}/imagesVal/{d_name}",
            "label": f"{data_path}/labelsVal/{d_name}"
        })

    train_transform = [
        LoadImaged(keys=["image", "scrib"]),
        AddChannel("scrib"),
        CropForegroundd(keys=["image", "scrib"], source_key="image"),
        Spacingd(
            keys=["image", "scrib"],
            pixdim=(1.5, 1.5, 1.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "scrib"], device="cuda", track_meta=False),
        Augment()
    ]
    train_ds = CacheDataset(
        data=train_data_list,
        transform=Compose(train_transform),
        cache_num=100,
        cache_rate=1.0,
        num_workers=16,
    )
    val_transform = [
        LoadImaged(keys=["image", "label"]),
        AddChannel("label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device="cuda", track_meta=False)
    ]
    val_ds = CacheDataset(
        data=val_data_list,
        transform=Compose(val_transform),
        cache_num=20,
        cache_rate=1.0,
        num_workers=16
    )
    return train_ds, val_ds
