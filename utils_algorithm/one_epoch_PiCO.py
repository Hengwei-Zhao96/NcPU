import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter, accuracy


def train_PiCO(args, train_loader, model, loss_cls_fn, loss_cont_fn, optimizer, epoch, tb_logger, start_upd_prot=False):
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    acc_cls = AverageMeter("Acc@Cls", ":2.2f")
    acc_proto = AverageMeter("Acc@Proto", ":2.2f")
    loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
    loss_cont_log = AverageMeter("Loss@Cont", ":2.2f")
    prototype_dist_log = AverageMeter("Dist@Proto", ":2.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log, prototype_dist_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))
    log_str_list = []

    ##################
    y_pred = []
    y_true_l = []
    y_l = []
    y_prot_score = []
    conf = []
    ##################

    # switch to train mode
    model.train()

    end = time.time()
    for i, (index, images_w, images_s, labels, true_labels) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()  # for showing training accuracy and will not be used when training

        cls_out, features_cont, pseudo_target_cont, score_prot = model(X_w, X_s, Y, args)
        # cls_out: classification logits without softmax
        # features_cont: features pool with L2 normalization
        # pseudo_target: class index for the features pool
        # score_prot: protopical logits without softmax
        
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        if start_upd_prot:
            loss_cls_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()  # get positive set by contrasting predicted labels
        else:
            mask = None
            # Warmup using MoCo

        ##################
        y_pred.append(torch.softmax(cls_out, dim=1)[:, 0])
        y_true_l.append(Y_true)
        y_l.append(Y)
        y_prot_score.append(score_prot[:, 0])
        conf.append(loss_cls_fn.confidence[index])
        ##################

        # contrastive loss
        loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        # classification loss
        loss_cls = loss_cls_fn(cls_out, index, epoch)

        loss = loss_cls + args.loss_weight * loss_cont
        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        proto_dist = F.cosine_similarity(model.module.prototypes[0],model.module.prototypes[1],dim=0)
        prototype_dist_log.update(proto_dist.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)
        if acc is not None:
            acc_cls.update(acc[0].item())
        acc = accuracy(score_prot, Y_true)
        if acc is not None:
            acc_proto.update(acc[0].item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i != 0 and i % (len(train_loader) - 1) == 0:
            log_str_list.append(progress.display(i + 1))

    ##################
    y_pred = torch.cat(y_pred, dim=0)
    y_true_l = torch.cat(y_true_l, dim=0)
    y_l = torch.cat(y_l, dim=0)
    conf = torch.cat(conf, dim=0)
    y_prot_score = torch.cat(y_prot_score, dim=0)

    p_index = torch.where(y_l[:, 1] == 0)[0]
    u_index = torch.where(y_l[:, 1] == 1)[0]
    u_n_index = torch.where(y_true_l == 1)[0]
    u_p_index = torch.where((y_l[:, 1] + y_true_l) == 1)[0]

    pred_p = y_pred[p_index].mean().detach().cpu()
    pred_u = y_pred[u_index].mean().detach().cpu()
    pred_up = y_pred[u_p_index].mean().detach().cpu()
    pred_un = y_pred[u_n_index].mean().detach().cpu()

    conf_u_n = conf[u_index, 1].mean().detach().cpu()
    conf_p_p = conf[p_index, 0].mean().detach().cpu()
    conf_up_p = conf[u_p_index, 0].mean().detach().cpu()
    conf_un_n = conf[u_n_index, 1].mean().detach().cpu()

    pred_prot_p = y_prot_score[p_index].mean().detach().cpu()
    pred_prot_up = y_prot_score[u_p_index].mean().detach().cpu()
    pred_prot_un = y_prot_score[u_n_index].mean().detach().cpu()
    ##################

    if args.gpu == 0:
        tb_logger.add_scalar("Cls Acc", acc_cls.avg, epoch)
        tb_logger.add_scalar("Prototype Acc", acc_proto.avg, epoch)
        tb_logger.add_scalar("Classification Loss", loss_cls_log.avg, epoch)
        tb_logger.add_scalar("Contrastive Loss", loss_cont_log.avg, epoch)
        tb_logger.add_scalar("Prototype Dist", prototype_dist_log.avg, epoch)

        tb_logger.add_scalar("Pred_P", pred_p.item(), epoch)
        tb_logger.add_scalar("Pred_U", pred_u.item(), epoch)
        tb_logger.add_scalar("Pred_UP", pred_up.item(), epoch)
        tb_logger.add_scalar("Pred_UN", pred_un.item(), epoch)

        tb_logger.add_scalar("Pred_Prot_P", pred_prot_p.item(), epoch)
        tb_logger.add_scalar("Pred_Prot_UP", pred_prot_up.item(), epoch)
        tb_logger.add_scalar("Pred_Prot_UN", pred_prot_un.item(), epoch)

        tb_logger.add_scalar("Conf_P_P", conf_p_p.item(), epoch)
        tb_logger.add_scalar("Conf_U_N", conf_u_n.item(), epoch)
        tb_logger.add_scalar("Conf_UP_P", conf_up_p.item(), epoch)
        tb_logger.add_scalar("Conf_UN_N", conf_un_n.item(), epoch)

    return log_str_list


def validate_PiCO(args, epoch, model, test_loader, tb_logger):

    print("==> Evaluation...")
    y_pred = []
    y_score = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for batch_idx, (_, images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, args, eval_only=True)

            _, pred = torch.max(outputs, dim=1)

            y_pred.append(pred)
            y_score.append(torch.softmax(outputs, dim=1)[:, 0])
            y_true.append(labels)

    y_pred = torch.cat(y_pred)
    y_score = torch.cat(y_score)
    y_true = torch.cat(y_true)

    testing_metrics = pu_metric(y_true, y_pred, y_score, pos_label=args.pos_label)
    testing_prin = metric_prin(testing_metrics)

    print(testing_prin)
    tb_logger.add_scalar("Top1 Acc", testing_metrics["OA"].item(), epoch)

    return testing_metrics, testing_prin
