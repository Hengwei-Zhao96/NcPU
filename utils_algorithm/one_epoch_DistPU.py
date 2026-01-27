import time
import logging

import torch

from utils_algorithm.DistPU import mixup_two_targets
from utils_loss_function.DistPU import loss_entropy, mixup_bce
from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter


def train_DistPU_warmup(args, data_loader, model, loss_fn, optimizer, schedular, epoch):
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time, loss_cls_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.warm_up_epochs + args.pu_epochs))

    model.train()

    end = time.time()
    for i, (index, Xs, Ys) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        Xs = Xs.cuda()
        Ys = Ys.cuda()
        outputs = model(Xs).squeeze()
        loss = loss_fn(outputs, Ys.float())

        loss_cls_log.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(data_loader) - 1) == 0:
            logging.info(progress.display(i + 1))

    schedular.step()
    args.tb_logger.add_scalar("Warmup Classification Loss", loss_cls_log.avg, epoch)


def train_DistPU_mixup(args, train_loader, model, mixup_dataset, base_loss, co_entropy, optimizer, schedular, epoch):
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_cls_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.warm_up_epochs + args.pu_epochs))

    model.train()
    end = time.time()
    for i, (index, Xs, Ys) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        Xs = Xs.cuda()
        Ys = Ys.cuda()
        psudos = mixup_dataset.psudo_labels[index].cuda()
        psudos[Ys == 1] = 1

        mixed_x, y_a, y_b, lam = mixup_two_targets(Xs, psudos, args.alpha)
        outputs = model(mixed_x).squeeze()
        outputs = torch.clamp(outputs, min=-10, max=10)
        scores = torch.sigmoid(outputs)

        outputs_ = torch.clamp(model(Xs).squeeze(), min=-10, max=10)
        scores_ = torch.sigmoid(outputs_)

        loss = (
            base_loss(outputs_, Ys.float())
            + co_entropy * loss_entropy(scores_[Ys != 1])
            + args.co_mix_entropy * loss_entropy(scores)
            + args.co_mixup * mixup_bce(scores, y_a, y_b, lam)
        )

        loss_cls_log.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mixup_dataset.psudo_labels[index] = scores_.detach()

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(train_loader) - 1) == 0:
            logging.info(progress.display(i + 1))

    schedular.step()
    args.tb_logger.add_scalar("Classification Loss", loss_cls_log.avg, epoch)


def validate_DistPU(args, epoch, data_loader, model):

    print("==> Evaluation...")
    y_pred = []
    y_score = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for _, (index, Xs, Ys) in enumerate(data_loader):
            Xs = Xs.cuda()
            Ys = Ys.cuda()
            outputs = model(Xs).view_as(Ys)

            y_pred.append(torch.where(outputs > 0, 1, 0))
            y_score.append(torch.sigmoid(outputs))
            y_true.append(Ys)

    y_pred = torch.cat(y_pred)
    y_score = torch.cat(y_score)
    y_true = torch.cat(y_true)

    testing_metrics = pu_metric(y_true, y_pred, y_score, pos_label=args.pos_label)
    testing_prin = metric_prin(testing_metrics)
    logging.info(testing_prin)
    print(testing_prin)
    args.tb_logger.add_scalar("Top1 Acc", testing_metrics["OA"].item(), epoch)
    return testing_metrics
