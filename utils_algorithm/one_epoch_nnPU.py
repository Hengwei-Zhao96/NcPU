import logging
import time

import torch

from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter


def train_nnPU(args, train_loader, model, loss_fn, optimizer, epoch):
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_cls_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (index, images, labels, true_labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, labels, index = images.cuda(), labels.cuda(), index.cuda()
        true_labels = true_labels.long().detach().cuda()  # for showing training accuracy and will not be used when training

        cls_out = model(images)

        # classification loss
        loss = loss_fn(cls_out, labels)

        loss_cls_log.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(train_loader) - 1) == 0:
            logging.info(progress.display(i + 1))

    args.tb_logger.add_scalar("Classification Loss", loss_cls_log.avg, epoch)


def validate_nnPU(args, epoch, model, test_loader):

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
