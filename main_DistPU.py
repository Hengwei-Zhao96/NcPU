import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader


from toolbox import pre_setting, save_checkpoint, str2lit
from utils_algorithm.DistPU import update_co_entropy
from utils_algorithm.one_epoch_DistPU import train_DistPU_warmup, train_DistPU_mixup, validate_DistPU
from utils_data.get_distpu_dataloader import get_distpu_dataloader, MixupDataset
from utils_loss_function.DistPU import DistPULoss

from utils_model.resnet import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: DistPU")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=1, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=256, type=int, help="mini-batch size (default: 256)")
    # Configuration of loss function
    parser.add_argument("--class_prior", type=float, default=0.4, help="class prior in loss function")
    # Configuration of optimization
    parser.add_argument("--warm_up_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr", default=5e-5, type=float, help="initial learning rate")
    parser.add_argument("--warm_up_weight_decay", type=float, default=5e-3)
    parser.add_argument("--weight_decay", default=1e-3, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--warm_up_epochs", type=int, default=60)
    parser.add_argument("--pu_epochs", type=int, default=60)
    # Configuration of DistPU
    parser.add_argument("--entropy", type=int, default=1, choices=[0, 1])
    parser.add_argument("--co_mu", type=float, default=2e-3, help="coefficient of L_ent")
    parser.add_argument("--co_entropy", type=float, default=0.004)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--co_mix_entropy", type=float, default=0.04)
    parser.add_argument("--co_mixup", type=float, default=5.0)
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "lr_{lr}_warm_up_epochs_{wep}_pu_epochs_{puep}_class_prior_{class_prior}_seed_{seed}".format(
        lr=args.lr, wep=args.warm_up_epochs, puep=args.pu_epochs, class_prior=str(args.class_prior), seed=args.seed
    )
    args = pre_setting(args, model_name="DistPU", model_path=model_path)
    #############################################

    train_loader, test_loader, mixup_loader = get_distpu_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    model = ResNet18(nb_classes=1,dataset_name=args.dataset).cuda()
    distpu_loss_warm_up = DistPULoss(args=args)
    if args.dataset=="stl10":
        optimizer_warm_up = torch.optim.SGD(model.parameters(), lr=args.warm_up_lr, weight_decay=args.warm_up_weight_decay)
    else:
        optimizer_warm_up = torch.optim.Adam(model.parameters(), lr=args.warm_up_lr, weight_decay=args.warm_up_weight_decay)
    schedular_warm_up = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warm_up, args.pu_epochs)

    print("\n==> Start Warmup Training...\n")
    best_acc = 0
    for epoch in range(args.warm_up_epochs):
        is_best = False

        train_DistPU_warmup(args=args, data_loader=train_loader, model=model, loss_fn=distpu_loss_warm_up, optimizer=optimizer_warm_up, schedular=schedular_warm_up, epoch=epoch)

        testing_metrics = validate_DistPU(args=args, epoch=epoch, model=model, data_loader=test_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer_warm_up.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )

    mixup_dataset = MixupDataset()
    mixup_dataset.update_psudos(mixup_loader, model=model)

    args.entropy = 0
    distpu_loss = DistPULoss(args=args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs, 0.7 * args.lr)

    print("\n==> Start Training...\n")
    for epoch in range(args.pu_epochs):
        is_best = False
        co_entropy = update_co_entropy(args=args, epoch=epoch)

        train_DistPU_mixup(
            args=args,
            train_loader=train_loader,
            model=model,
            mixup_dataset=mixup_dataset,
            base_loss=distpu_loss,
            co_entropy=co_entropy,
            optimizer=optimizer,
            schedular=schedular,
            epoch=args.warm_up_epochs + epoch,
        )

        testing_metrics = validate_DistPU(args=args, epoch=args.warm_up_epochs + epoch, model=model, data_loader=test_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": args.warm_up_epochs + epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )


if __name__ == "__main__":
    args = Argparse()
    main(args)
