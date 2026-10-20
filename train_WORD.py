import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from monai.data import ThreadDataLoader, set_track_meta
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss, Parameter
from tqdm import tqdm

from dataset_loader_WORD import get_dataset_WORD
from utils import test_util
from utils.losses import pDLoss
from utils.ood import ood_compute
from vnet import VNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='WORD', help='dataset_name')
parser.add_argument('--exp', type=str, default='ours', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=17, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=tuple, default=(96, 96, 96), help='patch size of network input')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--cuda', type=str, default="1", help='cuda id')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    model = VNet(n_classes=num_classes).cuda()

    train_ds, val_ds = get_dataset_WORD()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.batch_size,
                                    shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    set_track_meta(False)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    p_ce_loss_fn = CrossEntropyLoss(ignore_index=num_classes)
    p_dice_loss_fn = pDLoss(n_classes=num_classes, ignore_index=num_classes)
    ce_loss_fn = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_loader)))

    iter_num = 0
    best_dice_avg = 0
    num_interactive = 2
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    mean = Parameter(torch.randn(num_classes, 1), requires_grad=True).cuda()
    log_var = Parameter(torch.randn(num_classes, 1), requires_grad=True).cuda()
    log_var.data.clamp_min_(-10)
    mean_aux = Parameter(torch.randn(num_classes, 1), requires_grad=True).cuda()
    log_var_aux = Parameter(torch.randn(num_classes, 1), requires_grad=True).cuda()
    log_var_aux.data.clamp_min_(-10)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):

            volume_batch, scrib_batch = sampled_batch['image'], sampled_batch['scrib']
            scrib_batch_ce = scrib_batch.squeeze(1).long()
            bs = volume_batch.shape[0]

            last_map = torch.zeros((bs, 2, 96, 96, 96), dtype=torch.float32, device="cuda")

            for interactive in range(num_interactive):
                last_map_this = last_map.clone()
                in_x = torch.cat((volume_batch, last_map_this), dim=1)
                outputs, features, outputs_aux, features_aux = model(in_x)

                # pCE + CPS ============================================================================================
                outputs_soft1 = torch.softmax(outputs, dim=1)
                outputs_soft2 = torch.softmax(outputs_aux, dim=1)

                loss_sup1 = p_ce_loss_fn(outputs, scrib_batch_ce)
                loss_sup2 = p_ce_loss_fn(outputs_aux, scrib_batch_ce) + p_dice_loss_fn(outputs_aux, scrib_batch)
                loss_sup = 0.5 * (loss_sup1 + loss_sup2)  # 式3

                alpha = random.random() + 1e-10

                soft_pseudo_label = alpha * outputs_soft1.detach() + (1.0 - alpha) * outputs_soft2.detach()
                loss_pse_sup_soft = \
                    0.5 * (ce_loss_fn(outputs_soft1, soft_pseudo_label) + ce_loss_fn(outputs_soft2, soft_pseudo_label))
                pseudo_label = soft_pseudo_label.argmax(1).detach()

                # ======================================================================================================

                # 分支1 概率分布
                loss_ma, ood_map = ood_compute(features, pseudo_label, mean, log_var, num_classes)

                # 分支2 概率分布
                loss_ma_aux, ood_map_aux = ood_compute(features_aux, pseudo_label, mean_aux, log_var_aux, num_classes)

                ood_mask = (ood_map > ood_map_aux).int()
                ood_map_mix = ood_map * ood_mask + ood_map_aux * (1 - ood_mask)

                last_map[:, 0, ...] = ood_map_mix.detach()
                last_map[:, 1, ...] = pseudo_label.detach()

                # ======================================================================================================

                loss_ma = loss_ma + loss_ma_aux
                loss = loss_sup + 8.0 * loss_pse_sup_soft + 0.01 * loss_ma

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/loss', loss.clone().detach().cpu().numpy(), iter_num)
                writer.add_scalar('info/loss_sup1', loss_sup1.clone().detach().cpu().numpy(), iter_num)
                writer.add_scalar('info/loss_sup2', loss_sup2.clone().detach().cpu().numpy(), iter_num)
                writer.add_scalar('info/loss_sup', loss_sup.clone().detach().cpu().numpy(), iter_num)
                writer.add_scalar('info/loss_sps', loss_pse_sup_soft.clone().detach().cpu().numpy(), iter_num)
                writer.add_scalar('info/loss_ma', loss_ma.clone().detach().cpu().numpy(), iter_num)

                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

                distribution_dict = {
                    'mean': mean,
                    'mean_aux': mean_aux,
                    'log_var': log_var,
                    'log_var_aux': log_var_aux
                }
                if iter_num % 20 == 0 and interactive == (num_interactive - 1):
                    b, c, h, w, l = volume_batch.shape
                    ll = l // 2
                    image = volume_batch[0, :, :, :, ll]
                    image = (image - image.min()) / (image.max() - image.min())
                    image = image.cpu().detach().numpy()
                    writer.add_image('train/Image', image, iter_num)
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    pred = preds[0, :, :, :, ll].cpu().detach().numpy().astype(np.uint8) * 18
                    writer.add_image('train/Prediction', pred, iter_num)
                    preds_aux = torch.argmax(torch.softmax(outputs_aux, dim=1), dim=1, keepdim=True)
                    pred_aux = preds_aux[0, :, :, :, ll].cpu().detach().numpy().astype(np.uint8) * 15
                    writer.add_image('train/Prediction_aux', pred_aux, iter_num)
                    scrib = scrib_batch[0, :, :, :, ll].cpu().detach().numpy().astype(np.uint8) * 15
                    writer.add_image('train/Scrib', scrib, iter_num)

                    ood_map_np = ood_map[0, :, :, ll].cpu().detach().unsqueeze(0).numpy()
                    ood_map_np = (ood_map_np - ood_map_np.min()) / (ood_map_np.max() - ood_map_np.min())
                    writer.add_image('train/OOD_map', ood_map_np, iter_num)
                    ood_map_aux_np = ood_map_aux[0, :, :, ll].cpu().detach().unsqueeze(0).numpy()
                    ood_map_aux_np = (ood_map_aux_np - ood_map_aux_np.min()) / (ood_map_aux_np.max() - ood_map_aux_np.min())
                    writer.add_image('train/OOD_map_aux', ood_map_aux_np, iter_num)

                if iter_num > 0 and iter_num % 100 == 0 and interactive == (num_interactive - 1):
                    model.eval()
                    dice_all = test_util.validation_all_case_monai(model,
                                                                   num_classes=num_classes,
                                                                   val_loader=val_loader,
                                                                   patch_size=args.patch_size)
                    dice_avg = dice_all
                    logging.info('iteration {}, average model DSC: {:.3f}'.format(iter_num, dice_avg))
                    if dice_avg > best_dice_avg:
                        best_dice_avg = dice_avg
                        save_mode_path = os.path.join(snapshot_path,
                                                      'model_iter_{}_dice_{}.pth'.format(iter_num, best_dice_avg))
                        save_distribution_path = os.path.join(snapshot_path,
                                                      'distribution_iter_{}_dice_{}.pth'.format(iter_num, best_dice_avg))

                        torch.save(model.state_dict(), save_mode_path)
                        torch.save(distribution_dict, save_distribution_path)

                        save_best_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.exp))
                        save_best_distribution_path = os.path.join(snapshot_path, '{}_best_distribution.pth'.format(args.exp))
                        torch.save(model.state_dict(), save_best_model_path)
                        torch.save(distribution_dict, save_best_distribution_path)

                        logging.info("save best model to {}".format(save_mode_path))
                    writer.add_scalar('Val_dice/Dice', dice_avg, iter_num)
                    writer.add_scalar('Val_dice/Best_dice', best_dice_avg, iter_num)

                    logging.info('iteration %d : model_mean_dice : %f' % (iter_num, dice_avg))
                    model.train()

                if iter_num % 3000 == 0 and interactive == (num_interactive - 1):
                    save_mode_path = os.path.join(snapshot_path, 'model_iter_' + str(iter_num) + '.pth')
                    save_distribution_path = os.path.join(snapshot_path, 'distribution_iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(distribution_dict, save_distribution_path)
                    logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../{}_model/{}".format(args.exp, args.dataset_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
