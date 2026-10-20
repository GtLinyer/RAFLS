import torch


def ood_compute(features, pseudo_label, mean, log_var, num_classes, device="cuda"):
    b, c, w, h, l = features.shape

    std = torch.exp(0.5 * log_var)

    features_flat = features.view(b, c, -1)  # (b, c, w, h, l) -> (b, c, w*h*l)
    pseudo_label_flat = pseudo_label.view(b, -1)  # (b, w, h, l) -> (b, w*h*l)

    mahalanobis_sq = torch.zeros(1, device=device)
    ii = 0
    mahalanobis_map = torch.zeros((b, 1, w * h * l), device=device)
    for bb in range(b):
        features_batch = features_flat[bb]  # (c, w*h*l)
        pseudo_label_flat_batch = pseudo_label_flat[bb]  # (c, w*h*l)

        for cls in range(0, num_classes):
            mask = (pseudo_label_flat_batch == cls).int()  # (w*h*l)
            if mask.sum() > 0:
                features_batch_cls = features_batch.detach() * mask  # (c, w*h*l)

                mean_cls = mean[cls].expand(c, w * h * l) * mask  # (c, w*h*l)
                std_cls = std[cls].expand(c, w * h * l) * mask  # (c, w*h*l)

                diff = features_batch_cls - mean_cls  # (c, w*h*l)

                mahalanobis_sq_cls = ((diff ** 2) / (std_cls ** 2 + 10e-6)).sum(dim=0, keepdim=True) * mask  # (c, w*h*l)
                if cls != 0:
                    mahalanobis_map[bb] += mahalanobis_sq_cls
                mahalanobis_sq_cls = mahalanobis_sq_cls.sum() / mask.sum()

                # while True:
                #     cls_o = randint(0, num_classes-1)
                #     if cls_o != cls:
                #         break
                # mean_cls_o = mean[cls_o].expand(c, w * h * l) * mask  # (c, w*h*l)
                # std_cls_o = std[cls_o].expand(c, w * h * l) * mask  # (c, w*h*l)
                #
                # diff_o = features_batch_cls - mean_cls_o  # (c, w*h*l)
                #
                # mahalanobis_sq_cls_o = ((diff_o ** 2) / (std_cls_o ** 2 + 10e-6)).sum(dim=0, keepdim=True) * mask
                # mahalanobis_sq_cls_o = mahalanobis_sq_cls_o.sum() / mask.sum()
                #
                # mahalanobis_sq += (mahalanobis_sq_cls - 0.01 * mahalanobis_sq_cls_o)

                mahalanobis_sq += mahalanobis_sq_cls
                ii += 1
    mahalanobis_map = mahalanobis_map.view(b, w, h, l)
    loss_ma = mahalanobis_sq / ii
    return loss_ma, mahalanobis_map
