import argparse
import math

import numpy as np
import torch


from toolbox import pre_setting, str2lit
from utils_data.get_holisticpu_dataloader import get_holistic_dataloader
from utils_algorithm.HolisticPU import ModelEMA, get_cosine_schedule_with_warmup
from utils_algorithm.one_epoch_HolisticPU import train_HolisticPU_warmup, train_HolisticPU

from utils_model.resnet import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: HolisticPU")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size (default: 256)")
    # Configuration of optimization
    parser.add_argument("--lr", default=0.0015, type=float, help="initial learning rate")
    parser.add_argument("--nesterov", action="store_true", default=True, help="use nesterov momentum")
    parser.add_argument("--wd", "--weight_decay", default=5e-4, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--total_steps", default=25 * 2**9, type=int, help="number of total steps to run")
    parser.add_argument("--eval_step", default=512, type=int, help="number of eval steps to run")
    parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--warmup", default=0, type=float, help="warmup epochs (unlabeled data based)")
    parser.add_argument("--warming_steps", default=15 * 2**9, type=int, help="number of epochs in training phase 1")
    parser.add_argument("--rho", default=0.01, type=float, help="smoothing parameter")
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "seed_{seed}".format(seed=args.seed)
    args = pre_setting(args, model_name="HolisticPU", model_path=model_path)
    #############################################

    labeled_trainloader, unlabeled_trainloader, test_loader = get_holistic_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    no_decay = ["bias", "bn"]
    args.ft_epochs = args.total_steps // args.eval_step

    model_warmup = ResNet18(dataset_name=args.dataset).cuda()
    grouped_parameters_warmup = [
        {"params": [p for n, p in model_warmup.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model_warmup.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_warmup = torch.optim.SGD(grouped_parameters_warmup, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    scheduler_warnup = get_cosine_schedule_with_warmup(optimizer_warmup, args.warmup, args.total_steps)

    args.start_epoch = 0
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    args.warming_epochs = math.ceil(args.warming_steps / args.eval_step)
    ema_model = ModelEMA(args, model_warmup, 0.999)

    model_warmup.zero_grad()

    print("\n==> Start Warmup Training...\n")
    pseudo_targets = train_HolisticPU_warmup(
        args=args,
        labeled_trainloader=labeled_trainloader,
        unlabeled_trainloader=unlabeled_trainloader,
        model=model_warmup,
        optimizer=optimizer_warmup,
        ema_model=ema_model,
        scheduler=scheduler_warnup,
    )
    del model_warmup, optimizer_warmup, scheduler_warnup

    model = ResNet18(dataset_name=args.dataset).cuda()
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    model.zero_grad()

    print("\n==> Start Training...\n")
    train_HolisticPU(
        args=args,
        labeled_trainloader=labeled_trainloader,
        unlabeled_trainloader=unlabeled_trainloader,
        test_loader=test_loader,
        pseudo_targets=pseudo_targets,
        model1=model,
        optimizer1=optimizer,
        scheduler1=scheduler,
    )


if __name__ == "__main__":
    args = Argparse()
    main(args)
