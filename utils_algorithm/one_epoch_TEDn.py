import time
import logging

import numpy as np
import torch

from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter


def train_TEDn_warmup(args, epoch, net, p_trainloader, u_trainloader, optimizer, criterion):

    # setup some utilities for analyzing performance
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_log = AverageMeter("Loss@Total", ":2.2f")
    progress = ProgressMeter(args.train_interval, [batch_time, data_time, loss_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.warm_start_epochs))

    net.train()
    end = time.time()
    for batch_idx in range(args.train_interval):

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        try:
            _, u_inputs, u_targets, u_true_targets = next(u_iter)
        except:
            u_iter = iter(u_trainloader)
            _, u_inputs, u_targets, u_true_targets = next(u_iter)

        try:
            _, p_inputs, p_targets, _ = next(p_iter)
        except:
            p_iter = iter(p_trainloader)
            _, p_inputs, p_targets, _ = next(p_iter)

        p_targets = p_targets.cuda()
        u_targets = u_targets.cuda()

        inputs = torch.cat((p_inputs, u_inputs), dim=0).cuda()
        targets = torch.cat((p_targets, u_targets), dim=0)

        outputs = net(inputs)

        p_outputs = outputs[: len(p_targets)]
        u_outputs = outputs[len(p_targets) :]

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)
        loss = (p_loss + u_loss) / 2.0
        loss.backward()
        optimizer.step()

        loss_log.update(loss.item())

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx != 0 and batch_idx % (args.train_interval - 1) == 0:
            logging.info(progress.display(batch_idx + 1))

    args.tb_logger.add_scalar("Warmup Classification Loss", loss_log.avg, epoch)


def train_TEDn(args, epoch, net, p_trainloader, u_trainloader, optimizer, criterion, keep_sample=None):

    # setup some utilities for analyzing performance
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_log = AverageMeter("Loss@Total", ":2.2f")
    progress = ProgressMeter(args.train_interval, [batch_time, data_time, loss_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    net.train()
    end = time.time()
    for batch_idx in range(args.train_interval):

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        try:
            u_index, u_inputs, u_targets, u_true_targets = next(u_iter)
        except:
            u_iter = iter(u_trainloader)
            u_index, u_inputs, u_targets, u_true_targets = next(u_iter)

        try:
            _, p_inputs, p_targets, _ = next(p_iter)
        except:
            p_iter = iter(p_trainloader)
            _, p_inputs, p_targets, _ = next(p_iter)

        u_idx = np.where(keep_sample[u_index.numpy()] == 1)[0]

        u_targets = u_targets[u_idx]

        p_targets = p_targets.cuda()
        u_targets = u_targets.cuda()

        u_inputs = u_inputs[u_idx]
        inputs = torch.cat((p_inputs, u_inputs), dim=0)
        targets = torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.cuda()

        outputs = net(inputs)

        p_outputs = outputs[: len(p_targets)]
        u_outputs = outputs[len(p_targets) :]

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)

        loss = (p_loss + u_loss) / 2.0

        loss.backward()
        optimizer.step()

        loss_log.update(loss.item())

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx != 0 and batch_idx % (args.train_interval - 1) == 0:
            logging.info(progress.display(batch_idx + 1))

    args.tb_logger.add_scalar("Classification Loss", loss_log.avg, epoch)


def validate_TEDn(args, epoch, model, test_loader):

    print("==> Evaluation...")
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
