import time
import logging

import jenkspy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from toolbox import pu_metric, metric_prin, AverageMeter, save_checkpoint
from utils_algorithm.HolisticPU import interleave, de_interleave, three_sigma, loss_ft


def record(args, valid_loader, model, epoch):
    batch_time = AverageMeter("Time", ":1.2f")
    data_time = AverageMeter("Data", ":1.2f")

    end = time.time()
    valid_loader = tqdm(valid_loader)

    with torch.no_grad():
        preds = np.array(args.batch_size * 2)
        for batch_idx, (_, inputs, inputs_s, target_u, target) in enumerate(valid_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            target = target.to(args.device)
            outputs = model(inputs)
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(outputs)
            pred = np.array((pred[:, 0]).cpu())
            target = target.cpu()
            if batch_idx == 0:
                preds = pred
                targets = target
            else:
                preds = np.concatenate([preds, pred], axis=0)
                targets = np.concatenate([targets, target], axis=0)
            batch_time.update(time.time() - end)
            end = time.time()

            valid_loader.set_description(
                "Record Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. ".format(
                    batch=batch_idx + 1,
                    iter=len(valid_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                )
            )
        valid_loader.close()
    return preds, targets


def train_HolisticPU_warmup(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, scheduler):

    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    model.train()
    for epoch in range(args.start_epoch, args.warming_epochs):
        batch_time = AverageMeter("Time", ":1.2f")
        data_time = AverageMeter("Data", ":1.2f")
        losses = AverageMeter("Loss@Total", ":2.2f")
        losses_x = AverageMeter("Loss@P", ":2.2f")
        losses_n = AverageMeter("Loss@U", ":2.2f")

        p_bar = tqdm(range(args.eval_step))

        for batch_idx in range(args.eval_step):
            try:
                _, inputs_x_w, inputs_x_s, _, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                _, inputs_x_w, inputs_x_s, _, targets_x = next(labeled_iter)

            try:
                _, inputs_u_w, inputs_u_s, targets_u, targets_t = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                _, inputs_u_w, inputs_u_s, targets_u, targets_t = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x_w.shape[0]
            inputs = interleave(torch.cat((inputs_u_w, inputs_u_s, inputs_x_w, inputs_x_s)), 4).cuda()
            targets_x = targets_x.cuda()
            targets_u = targets_u.cuda()
            logits = model(inputs)
            logits = de_interleave(logits, 4)
            logits_u, logits_u_w = logits[: 2 * batch_size].chunk(2)
            logits_x_w, logits_x_s = logits[2 * batch_size :].chunk(2)
            del logits

            Lx = (
                F.cross_entropy(logits_x_w, targets_x, reduction="mean", label_smoothing=args.rho)
                + F.cross_entropy(logits_x_s, targets_x, reduction="mean", label_smoothing=args.rho)
            ) / 2

            Ln = F.cross_entropy(logits_u, targets_u, reduction="mean")

            loss = Lx + Ln

            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_n.update(Ln.item())
            optimizer.step()
            scheduler.step()

            ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_n: {loss_n:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.warming_epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_n=losses_n.avg,
                )
            )
            p_bar.update()

        p_bar.close()

        test_model = ema_model.ema

        # Validation & Test
        preds, targets = record(args, unlabeled_trainloader, test_model, epoch)
        if epoch == 0:
            preds_sequence = preds
        else:
            preds_sequence = np.vstack((preds_sequence, preds))

    preds_sequence = preds_sequence.T
    trends = np.zeros(len(preds_sequence))
    for i, sequence in enumerate(preds_sequence):
        sequence = pd.Series(sequence)
        diff_1 = sequence.diff(periods=1)
        diff_1 = np.array(diff_1)
        diff_1 = diff_1[1:]
        diff_1 = np.log(1 + diff_1 + 0.5 * diff_1**2)
        trends[i] = diff_1.mean()
    intervals = jenkspy.jenks_breaks(trends, n_classes=2)
    break_point = intervals[1]
    if break_point > 0:
        trends_std = three_sigma(trends)
        intervals = jenkspy.jenks_breaks(trends_std, n_classes=2)
        break_point = intervals[1]
    logging.info(f"The interval is {intervals}; Break Point is {break_point}")
    pseudo_targets = np.where(trends > break_point, 0, 1)
    estimated_prior = 1 - pseudo_targets.sum() / len(pseudo_targets)
    prit_class_prior = "Estimated positive prior: " + str(estimated_prior)
    logging.info(prit_class_prior)
    print(prit_class_prior)
    return pseudo_targets


def validate_HolisticPU(args, epoch, model, test_loader):

    y_pred = []
    y_score = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for batch_idx, (_, images, labels) in enumerate(test_loader):
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


def train_HolisticPU(args, labeled_trainloader, unlabeled_trainloader, test_loader, pseudo_targets, model1, optimizer1, scheduler1):
    end = time.time()
    unlabeled_num = len(pseudo_targets)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    model1.train()
    unlabeled_idx = 0

    best_acc = 0
    for epoch in range(args.start_epoch, args.ft_epochs):
        is_best = False
        batch_time = AverageMeter("Time", ":1.2f")
        data_time = AverageMeter("Data", ":1.2f")
        losses = AverageMeter("Loss@Total", ":2.2f")
        losses_x = AverageMeter("Loss@P", ":2.2f")
        losses_u = AverageMeter("Loss@N", ":2.2f")
        p_bar = tqdm(range(args.eval_step))
        for batch_idx in range(args.eval_step):
            try:
                _, inputs_x_w, inputs_x_s, _, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                _, inputs_x_w, inputs_x_s, _, targets_x = next(labeled_iter)

            try:
                _, inputs_u_w, inputs_u_s, targets_u, targets_t = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                _, inputs_u_w, inputs_u_s, targets_u, targets_t = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x_w.shape[0]

            inputs = interleave(torch.cat((inputs_u_w, inputs_u_s, inputs_x_w, inputs_x_s)), 4).to(args.device)

            targets_x = targets_x.to(args.device)
            targets_u = targets_u.to(args.device)
            targets_t = targets_t.to(args.device)
            targets_p = pseudo_targets[unlabeled_idx : unlabeled_idx + batch_size]
            targets_p = torch.tensor(targets_p)
            targets_p = targets_p.to(torch.long).to(args.device)
            unlabeled_idx = (unlabeled_idx + batch_size) % unlabeled_num

            logits1 = model1(inputs)
            logits1 = de_interleave(logits1, 4)
            logits1_u, logits1_u_s = logits1[: 2 * batch_size].chunk(2)
            logits1_x_w, logits1_x_s = logits1[2 * batch_size :].chunk(2)

            del logits1
            Lx1 = F.cross_entropy(logits1_x_w, targets_x, reduction="mean")

            Lu1 = loss_ft(args, logits1_u, logits1_u_s, targets_u, targets_p, epoch=epoch)

            loss1 = Lx1 + Lu1

            loss = loss1

            losses.update(loss.item())
            losses_x.update(Lx1.item())
            losses_u.update(Lu1.item())

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            scheduler1.step()

            model1.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.ft_epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler1.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
            p_bar.update()
        p_bar.close()

        testing_metrics = validate_HolisticPU(args=args, epoch=epoch, model=model1, test_loader=test_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model1.state_dict(), "optimizer": optimizer1.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )
