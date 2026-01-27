import logging
import time

import torch
import torch.nn.functional as F

from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter, accuracy
from utils_algorithm.NcPU import get_cont_mask, get_pun_index

def train_NcPU(args, train_loader, model, cls_loss_fn, cont_loss_fn, ent_loss_fn, optimizer, scaler, epoch):
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    acc_cls = AverageMeter("Acc@Cls", ":2.2f")
    acc_proto = AverageMeter("Acc@Proto", ":2.2f")
    acc_pse_n = AverageMeter("Acc@Pse_N", ":2.2f")
    acc_pse_p = AverageMeter("Acc@Pse_P", ":2.2f")
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    loss_cont_log = AverageMeter("Loss@Cont", ":2.2f")
    prototype_dist_log = AverageMeter("Dist@Proto", ":2.2f")
    pseudo_n_num = AverageMeter("#Pse_N", ":d", is_sum=True)
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, acc_cls, acc_proto, acc_pse_n, acc_pse_p, pseudo_n_num, loss_cls_log, loss_cont_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    ##################
    y_pred_list = []
    pse_n_pred_list= []
    true_lable_list = []
    lable_list = []
    y_prot_score_list = []
    confidence_list = []
    ##################

    # switch to train mode
    model.train()

    end = time.time()
    for i, (index, images_w, images_s_online, images_s_target, labels, true_labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        index = index.cuda(args.gpu, non_blocking=True)
        images_w, images_s_online, images_s_target = images_w.cuda(args.gpu, non_blocking=True), images_s_online.cuda(args.gpu, non_blocking=True), images_s_target.cuda(args.gpu, non_blocking=True)
        labels, true_labels = labels.cuda(args.gpu, non_blocking=True), true_labels.long().detach().cuda(args.gpu, non_blocking=True) # true_labels for showing training accuracy and will not be used when training

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            online_out, target_out, cls_logits, prototype_pro = model(images_w, images_s_online, images_s_target, labels=labels, args=args)

            model.threshold_cls_update(cls_logits, labels=labels)
            if epoch >= args.warm_up:
                model.confidence_update(prototype_pro, index, labels, conf_ema_m=args.conf_ema_m)

            # classification loss
            idxs = get_pun_index(cls_logits, labels, threshold=model.threshold_param, softmax=True)
            loss_cls = cls_loss_fn(cls_logits, confidence=model.label_conf[index, :], idxs=idxs, epoch=epoch)

            pse_n_pred_list.append(torch.softmax(cls_logits, dim=1)[:, 1][idxs["pse_n_idx"]])
        
            mask = get_cont_mask(args.tau, cls_logits, epoch=epoch, warm_up_epoch=args.warm_up)
            loss_cont = cont_loss_fn(online_out, target_out, mask)

            # entropy loss
            loss_ent = ent_loss_fn(cls_logits)
            ent_loss_weight = min((epoch / args.warm_up) * args.ent_loss_weight, args.ent_loss_weight)

            loss = loss_cls + args.cont_loss_weight * loss_cont + ent_loss_weight * loss_ent
        
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=30)
        scaler.step(optimizer)
        scaler.update()

        ##################
        y_pred_list.append(torch.softmax(cls_logits, dim=1)[:, 1])
        true_lable_list.append(true_labels)
        lable_list.append(labels)
        y_prot_score_list.append(prototype_pro[:, 1])
        confidence_list.append(model.label_conf[index][:, 1])
        ##################

        pseudo_n_num.update(len(idxs["pse_n_idx"]))
        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        proto_dist = F.cosine_similarity(model.prototypes[0],model.prototypes[1],dim=0)
        prototype_dist_log.update(proto_dist.item())

        acc = accuracy(cls_logits, true_labels)
        if acc is not None:
            acc_cls.update(acc[0].item())
        acc = accuracy(prototype_pro, true_labels)
        if acc is not None:
            acc_proto.update(acc[0].item())
        
        pse_label_n = torch.stack((torch.zeros(len(idxs["pse_n_idx"])),torch.ones(len(idxs["pse_n_idx"]))), dim=1).cuda()
        pse_true_label = true_labels[idxs["pse_n_idx"]]
        acc = accuracy(pse_label_n, pse_true_label)
        if acc is not None:
            acc_pse_n.update(acc[0].item())

        pse_label_p = torch.stack((torch.ones(len(idxs["pse_p_idx"])),torch.zeros(len(idxs["pse_p_idx"]))), dim=1).cuda()
        pse_true_label_p = true_labels[idxs["pse_p_idx"]]
        acc = accuracy(pse_label_p, pse_true_label_p)
        if acc is not None:
            acc_pse_p.update(acc[0].item())

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(train_loader) - 1) == 0:
            logging.info(progress.display(i + 1))

    ##################
    y_pred_list = torch.cat(y_pred_list, dim=0)
    true_lable_list = torch.cat(true_lable_list, dim=0)
    lable_list = torch.cat(lable_list, dim=0)
    confidence_list = torch.cat(confidence_list, dim=0)
    y_prot_score_list = torch.cat(y_prot_score_list, dim=0)
    pse_n_pred_list = torch.cat(pse_n_pred_list, dim=0).mean()

    p_index = torch.where(lable_list == 0)[0]
    u_n_index = torch.where(true_lable_list == 1)[0]
    u_p_index = torch.where((lable_list == 1) & (true_lable_list == 0))[0]

    pred_pro_p = y_pred_list[p_index].mean()
    pred_pro_up = y_pred_list[u_p_index].mean()
    pred_pro_un = y_pred_list[u_n_index].mean()

    conf_p = confidence_list[p_index].mean()
    conf_u_p = confidence_list[u_p_index].mean()
    conf_u_n = confidence_list[u_n_index].mean()

    prot_p = y_prot_score_list[p_index].mean()
    prot_up = y_prot_score_list[u_p_index].mean()
    prot_un = y_prot_score_list[u_n_index].mean()

    args.tb_logger.add_scalar("Cls_Pro->P", pred_pro_p.item(), epoch)
    args.tb_logger.add_scalar("Cls_Pro->U(P)", pred_pro_up.item(), epoch)
    args.tb_logger.add_scalar("Cls_Pro->U(N)", pred_pro_un.item(), epoch)
    args.tb_logger.add_scalar("Cls_Pro->U(pse_N)", pse_n_pred_list.item(), epoch)

    args.tb_logger.add_scalar("Prot_Pro->P", prot_p.item(), epoch)
    args.tb_logger.add_scalar("Prot_Pro->U(P)", prot_up.item(), epoch)
    args.tb_logger.add_scalar("Prot_Pro->U(N)", prot_un.item(), epoch)

    args.tb_logger.add_scalar("Confidence->P", conf_p.item(), epoch)
    args.tb_logger.add_scalar("Confidence->U(P)", conf_u_p.item(), epoch)
    args.tb_logger.add_scalar("Confidence->U(N)", conf_u_n.item(), epoch)
    ##################
    
    args.tb_logger.add_scalar("Loss->Classification", loss_cls_log.avg, epoch)
    args.tb_logger.add_scalar("Loss->Noisy Contrastive", loss_cont_log.avg, epoch)
    args.tb_logger.add_scalar("Prototype Dist", prototype_dist_log.avg, epoch)
    args.tb_logger.add_scalar("Training Acc->Classifier", acc_cls.avg, epoch)
    args.tb_logger.add_scalar("Training Acc->Prototype", acc_proto.avg, epoch)
    args.tb_logger.add_scalar("Threshold->Positive", model.threshold_param[0], epoch)
    args.tb_logger.add_scalar("Threshold->Negative", model.threshold_param[1], epoch)

def validate_NcPU(args, epoch, model, test_loader):

    print("==> Evaluation...")
    y_pred = []
    y_score = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for batch_idx, (_, images, labels) in enumerate(test_loader):
            images, labels = images.cuda(args.gpu, non_blocking=True), labels.cuda(args.gpu, non_blocking=True)
            outputs = model(images, eval_only=True)

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
