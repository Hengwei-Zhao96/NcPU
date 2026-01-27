import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils_algorithm.LaGAM import create_model
from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter, accuracy
from utils_model.LaGAM.meta_layers import to_var


def train_LaGAM(args, train_loader, valid_loader, model, optimizer, bce_loss, contrastive_loss, epoch, cluster_result=None):

    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    acc_cls = AverageMeter("Acc@Cls", ":2.2f")
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    loss_cont_log = AverageMeter("Loss@Cont", ":2.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, acc_cls, loss_cls_log, loss_cont_log], prefix="Epoch: [{}]".format(epoch + 1))

    model.train()

    updated_label_list = []
    true_label_list = []
    index_list = []
    ema_param = 1.0 * epoch / args.epochs * (args.rho_end - args.rho_start) + args.rho_start

    end = time.time()

    for i, (images, images_s, labels_, true_labels, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if labels_.sum() == 0:
            continue
        true_label_list.append(true_labels)
        index_list.append(index)

        images, images_s, labels_, index = (
            images.cuda(),
            images_s.cuda(),
            labels_.cuda(),
            index.cuda(),
        )
        labels_ = labels_.unsqueeze(1)
        labels = torch.cat([1 - labels_, labels_], dim=1).detach()
        Y_true = true_labels.long().detach().cuda()
        bs = len(labels)
        cluster_idxes = None if cluster_result is None else cluster_result["im2cluster"][index]

        if epoch < args.warmup_epoch:
            labels_final = labels
        else:
            meta_model = create_model(num_class=1, dataset_name=args.dataset).cuda()
            meta_model.load_state_dict(model.state_dict())

            preds_meta = meta_model(images)

            eps = to_var(torch.zeros(bs, 2).cuda())
            labels_meta = labels + eps
            loss = bce_loss(preds_meta, labels_meta)

            meta_model.zero_grad()

            params = []
            for name, p in meta_model.named_params(meta_model):
                if args.identifier in name and len(p.shape) > 1:
                    params.append(p)
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            meta_lr = 0.001
            meta_model.update_params(meta_lr, source_params=grads, identifier=args.identifier)

            try:
                images_v, labels_v = next(valid_loder_iter)
            except:
                valid_loder_iter = iter(valid_loader)
                images_v, labels_v = next(valid_loder_iter)

            images_v = images_v.cuda()
            labels_v = F.one_hot(labels_v.cuda(), 2).float()

            preds_v = meta_model(images_v)

            loss_meta_v = bce_loss(preds_v, labels_v)
            grad_eps = torch.autograd.grad(loss_meta_v, eps, only_inputs=True, allow_unused=True)[0]

            eps = eps - grad_eps
            meta_detected_labels = eps.argmax(dim=1)
            meta_detected_labels[labels_.squeeze() == 1] = 1
            meta_detected_labels = F.one_hot(meta_detected_labels, 2)
            meta_detected_labels = meta_detected_labels.detach()

            updated_labels = labels
            updated_labels = updated_labels * ema_param + meta_detected_labels * (1 - ema_param)
            labels_final = updated_labels.detach()

            updated_label_list.append(updated_labels[:, 1].cpu())

            del grad_eps, grads, params

        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        X_w_c = images
        pseudo_label_c = labels_final
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix = model(X_w_c_mix)
        loss_mix = bce_loss(logits_mix, pseudo_label_c_mix)

        preds_final, feat_cont = model(images, flag_feature=True)
        loss_cls = bce_loss(preds_final, labels_final)

        loss_final = loss_cls + args.mix_weight * loss_mix

        _, feat_cont_s = model(images_s, flag_feature=True)
        loss_cont = contrastive_loss(
            feat_cont,
            feat_cont_s,
            cluster_idxes,
            preds_final,
            start_knn_aug=epoch > 50,
        )
        loss_final = loss_final + loss_cont

        loss_cont_log.update(loss_cont.item())
        loss_cls_log.update(loss_final.item())

        acc = accuracy(torch.cat([1 - preds_final, preds_final], dim=1), Y_true)
        if acc is not None:
            acc_cls.update(acc[0].item())

        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(train_loader) - 1) == 0:
            logging.info(progress.display(i + 1))

    if epoch >= args.warmup_epoch:
        true_label_list = torch.cat(true_label_list, dim=0)
        updated_label_list = torch.cat(updated_label_list, dim=0)
        index_list = torch.cat(index_list, dim=0)

        update_label_cate = (updated_label_list > 0.5) * 1
        compare = update_label_cate == true_label_list
        a = "New target accuracy: " + str(compare.sum() / len(compare))
        print(a)
        logging.info(a)

        train_loader.dataset.update_targets(updated_label_list.numpy(), index_list)


def validate_LaGAM(args, epoch, model, test_loader):

    print("==> Evaluation...")
    y_pred = []
    y_score = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            outputs = model(images)

            y_pred.append(torch.where(outputs > 0, 1, 0))
            y_score.append(torch.sigmoid(outputs))
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


def compute_features(model, eval_loader):
    model.eval()
    feat_list = torch.zeros(len(eval_loader.dataset), 128)
    with torch.no_grad():
        for i, (images, _, _, _, index) in enumerate(eval_loader):
            images = images.cuda(non_blocking=True)
            _, feat = model(images, flag_feature=True)
            feat_list[index] = feat.cpu()
    return feat_list.numpy()
