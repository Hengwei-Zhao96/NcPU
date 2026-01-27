import logging
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from toolbox import pu_metric, metric_prin
from utils_algorithm.WSC import AverageMeter
from utils_loss_function.WSC import ce_loss


def train_WSC(args, train_loader, model, noise_model, projector, optimizer, noise_matrix_optimizer, sup_loss, weak_spec_loss, scheduler, epoch):
    losses = AverageMeter()
    sup_losses = AverageMeter()
    vol_losses = AverageMeter()
    wsc_losses = AverageMeter()
    consist_losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()

    # switch to train mode
    model.train()
    noise_model.train()

    progress_bar = tqdm(enumerate(train_loader) ,total=len(train_loader), desc=f"Train Epoch: [{epoch + 1}]")

    for i, (x_w, x_s, x_s_, noise_y, index) in progress_bar:
        x_w = x_w.cuda()
        x_s = x_s.cuda()
        x_s_ = x_s_.cuda()
        noise_y = noise_y.cuda()

        x_aug = torch.cat([x_w, x_s, x_s_])
        y_pred, feat = model(x_aug)
        feat = projector(feat)
        feat = F.normalize(feat, dim=1)

        y_pred_w, y_pred_s, y_pred_s_ = y_pred.chunk(3)
        feat_w, feat_s, feat_s_ = feat.chunk(3)

        noise_matrix = noise_model()

        probs_x_w = y_pred_w.softmax(dim=-1).detach()

        noisy_probs_x_w = torch.matmul(y_pred_w.softmax(dim=-1), noise_matrix)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdim=True)

        # supervised loss
        supervised_loss = sup_loss(noisy_probs_x_w,noise_y)
        # supervised_loss = torch.mean(-torch.sum(F.one_hot(noise_y, 2) * torch.log(noisy_probs_x_w), dim = -1))

        # VolMinNet loss
        vol_loss = noise_matrix.slogdet().logabsdet

        # consistency loss
        con_loss = ce_loss(y_pred_s, probs_x_w, reduction='mean')
        con_loss_ = ce_loss(y_pred_s_, probs_x_w, reduction='mean')

        # wsc loss
        """
        Note that there's not one way to construct the wsc loss. you will also notice that there's a function called create_noise_matrix_inv in models.utils, which provide another way to do it. Our paper also mentioned that case and the experiments for this way (yes it actually using the enviroment information) is under exploration actually. An example is here:
            >>> true_noisy_matrix_inv = create_noise_matrix_inv(args.num_classes, args.noise_ratio).detach().cuda()
            >>> construced_s = (F.one_hot(y, args.num_classes).float() @ true_noisy_matrix_inv).detach()
            >>> wsc_loss, l1, l2 = weak_spec_loss(feat_s, feat_s_, constructed_s)
        For our implementation, we just use the probabilities of the model output as the constructed s.
        """
        wsc_loss, l1, l2 = weak_spec_loss(feat_s, feat_s_, probs_x_w)

        # total loss
        lam = min(1, float(epoch)/float(args.epochs)) * args.lam
        loss = supervised_loss + con_loss + con_loss_ + args.vol_lambda * vol_loss + lam * wsc_loss

        # compute average entropy loss
        if args.average_entropy_loss:
            avg_prediction = torch.mean(y_pred_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0 / 2 * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=1e-6, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = args.balance_lam * balance_kl
            loss += entropy_loss

         # update parameters
        loss.backward()

        optimizer.step()
        noise_matrix_optimizer.step()

        optimizer.zero_grad()
        noise_matrix_optimizer.zero_grad()

        scheduler.step()        

        # update meter
        losses.update(loss.item(), x_w.size(0))
        sup_losses.update((supervised_loss).item(), x_w.size(0))
        vol_losses.update(args.vol_lambda * vol_loss.item(), x_w.size(0))
        wsc_losses.update(wsc_loss.item(), x_w.size(0))
        consist_losses.update((con_loss).item(), x_w.size(0))
        l1_losses.update(l1.item(), x_w.size(0))
        l2_losses.update(l2.item(), x_w.size(0))

        progress_bar.set_postfix({
                "loss": f"{losses.val:.4f}",
                "sup": f"{sup_losses.val:.4f}",
                "vol": f"{vol_losses.val:.4f}",
                "wsc": f"{wsc_losses.val:.4f}",
                "con": f"{consist_losses.val:.4f}",
                "l1": f"{l1_losses.val:.4f}",
                "l2": f"{l2_losses.val:.4f}",
            })


def validate_WSC(args, epoch, model, test_loader):

    print("==> Evaluation...")
    y_pred = []
    y_score = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            _, pred = torch.max(outputs, dim=1)

            y_pred.append(pred)
            y_score.append(torch.softmax(outputs, dim=1)[:, 0])
            y_true.append(labels)

    y_pred = torch.cat(y_pred)
    y_score = torch.cat(y_score)
    y_true = torch.cat(y_true)

    testing_metrics = pu_metric(y_true, y_pred, y_score, pos_label=args.pos_label)
    testing_prin = metric_prin(testing_metrics)
    logging.info(testing_prin)
    print(testing_prin)
    args.tb_logger.add_scalar("Top1 Acc", testing_metrics["OA"].item(), epoch)

    return testing_metrics
