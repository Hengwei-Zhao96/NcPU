import logging
import math
import time

import torch

from toolbox import pu_metric, metric_prin, AverageMeter, ProgressMeter


def train_vPU(args, train_p_loader, train_u_loader, model, optimizer, epoch):

    # setup some utilities for analyzing performance
    batch_time = AverageMeter("Time", ":1.2f", is_sum=True)
    data_time = AverageMeter("Data", ":1.2f", is_sum=True)
    loss_log = AverageMeter("Loss@Total", ":2.2f")
    var_loss_log = AverageMeter("Loss@Var", ":2.2f")
    reg_loss_log = AverageMeter("Loss@Reg", ":2.2f")
    progress = ProgressMeter(args.val_iterations, [batch_time, data_time, loss_log, var_loss_log, reg_loss_log], prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs))

    # set the model to train mode
    model.train()
    end = time.time()
    for batch_idx in range(args.val_iterations):

        try:
            _, data_u, _, _ = next(u_iter)
        except:
            u_iter = iter(train_u_loader)
            _, data_u, _, _ = next(u_iter)

        try:
            _, data_p, _, _ = next(p_iter)
        except:
            p_iter = iter(train_p_loader)
            _, data_p, _, _ = next(p_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        data_p, data_u = data_p.cuda(), data_u.cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_u))
        output_phi_all = model(data_all)
        log_phi_all = output_phi_all[:, 0]
        idx_p = slice(0, len(data_p))
        idx_u = slice(len(data_p), len(data_all))
        log_phi_u = log_phi_all[idx_u]
        log_phi_p = log_phi_all[idx_p]
        output_phi_u = output_phi_all[idx_u]
        var_loss = torch.logsumexp(log_phi_u, dim=0) - math.log(len(log_phi_u)) - 1 * torch.mean(log_phi_p)

        # perform Mixup and calculate the regularization
        target_x = output_phi_u[:, 0].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda()
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(args.mix_alpha, args.mix_alpha)
        lam = m.sample()
        data = lam * data_u + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm

        data = data.cuda()
        target = target.cuda()
        out_log_phi_all = model(data)
        reg_mix_loss = ((torch.log(target) - out_log_phi_all[:, 0]) ** 2).mean()

        # calculate gradients and update the network
        loss = var_loss + args.lam * reg_mix_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the utilities for analysis of the model
        reg_loss_log.update(reg_mix_loss.item())
        loss_log.update(loss.item())
        var_loss_log.update(var_loss.item())

        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx != 0 and batch_idx % (args.val_iterations - 1) == 0:
            logging.info(progress.display(batch_idx + 1))

    args.tb_logger.add_scalar("Total Loss", loss_log.avg, epoch)
    args.tb_logger.add_scalar("Variational Loss", var_loss_log.avg, epoch)
    args.tb_logger.add_scalar("Regularization Loss", reg_loss_log.avg, epoch)


def cal_val_var(model, val_p_loader, val_u_loader):

    # set the model to evaluation mode
    model.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, (_, data_u, _, _) in enumerate(val_u_loader):
            data_u = data_u.cuda()
            output_phi_u_curr = model(data_u)
            if idx == 0:
                output_phi_u = output_phi_u_curr
            else:
                output_phi_u = torch.cat((output_phi_u, output_phi_u_curr))
        for idx, (_, data_p, _, _) in enumerate(val_p_loader):
            data_p = data_p.cuda()
            output_phi_p_curr = model(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
    log_phi_p = output_phi_p[:, 0]
    log_phi_u = output_phi_u[:, 0]
    var_loss = torch.logsumexp(log_phi_u, dim=0) - math.log(len(log_phi_u)) - torch.mean(log_phi_p)
    return var_loss.item()


def validate_vPU(args, model, train_p_loader, train_u_loader, test_loader, epoch):

    print("==> Evaluation...")

    # set the model to evaluation mode
    model.eval()

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (_, data, _, _) in enumerate(train_p_loader):
        data = data.cuda()
        log_max_phi = max(log_max_phi, model(data)[:, 0].max())
    for idx, (_, data, _, _) in enumerate(train_u_loader):
        data = data.cuda()
        log_max_phi = max(log_max_phi, model(data)[:, 0].max())

    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (_, data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            log_phi = model(data)[:, 0]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))

    testing_metrics = pu_metric(target_all, torch.where(log_phi_all > math.log(0.5), 0, 1), log_phi_all, pos_label=args.pos_label)
    testing_prin = metric_prin(testing_metrics)
    logging.info(testing_prin)
    print(testing_prin)
    args.tb_logger.add_scalar("Top1 Acc", testing_metrics["OA"].item(), epoch)

    return testing_metrics
