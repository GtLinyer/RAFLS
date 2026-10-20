import math

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import MetricReduction
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

from dataset_loader_BTCV import get_dataset_BTCV
from vnet import VNet


def validation_all_case_monai(model, num_classes, val_loader, sw_batch_size=4, patch_size=(96, 96, 96)):
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=False)
    for i_batch, sampled_batch in enumerate(val_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda().long()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                b, c, w, h, l = volume_batch.shape
                last_map = torch.zeros((1, 2, w, h, l), dtype=torch.float32, device="cuda")
                in_x = torch.cat((volume_batch, last_map), dim=1)
                prediction, _, _, _ = sliding_window_inference(inputs=in_x, roi_size=patch_size,
                                                               sw_batch_size=sw_batch_size, predictor=model)

        val_labels_list = decollate_batch(label_batch)
        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(prediction)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    all_metric = dice_metric.aggregate().item()
    return all_metric


def validation_all_case(model, num_classes, val_loader, patch_size=(96, 96, 96)):
    total_metric = []
    for i_batch, sampled_batch in enumerate(val_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.squeeze(0).squeeze(0).numpy()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = sliding_window_inference(inputs=volume_batch, roi_size=patch_size,
                                                sw_batch_size=1, predictor=model)
            prediction = pred.cpu().argmax(1).squeeze(0).numpy()
            if np.sum(prediction) == 0:
                case_metric = np.zeros((4, num_classes - 1))
            else:
                case_metric = np.zeros((4, num_classes - 1))
                for i in range(1, num_classes):
                    case_metric[:, i - 1] = cal_metric_val(prediction == i, label_batch == i)
            total_metric.append(np.expand_dims(case_metric, axis=0))

        # all_metric:22x4x8
        all_metric = np.concatenate(total_metric, axis=0)
        avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
        return avg_dice, std_dice, all_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # score_map : CxDxHxW, cnt: DxHxW
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    # score map: CxDxHxW, label_map: CxDxHxW -> DxHxW
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        sf = compute_surface_distances(gt, pred, spacing_mm=(1., 1., 1.))
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
        return np.array([dice, hd95, nsd, asd])
    else:
        return np.zeros(4)


def cal_metric_val(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return np.array([dice])
    else:
        return np.zeros(1)


if __name__ == '__main__':
    model = VNet(n_classes=14).cuda()
    train_d, val_d = get_dataset_BTCV()
    val_loader = ThreadDataLoader(val_d, num_workers=16, batch_size=1)
    avg_dice, std_dice, all_metric = validation_all_case(model, 14, val_loader, patch_size=(96, 96, 96))
    print(avg_dice.mean())
