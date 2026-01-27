import argparse

import numpy as np
import torch

from toolbox import pre_setting, adjust_learning_rate, save_checkpoint, str2lit
from utils_algorithm.one_epoch_CE import train_CE, validate_CE
from utils_data.get_ce_dataloader import get_ce_dataloader
from utils_loss_function.CE import CELoss
from utils_model.resnet import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: Cross Entropy")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name") # "cifar10"
    parser.add_argument("--pos_label", type=int, default=0, help="positive label") # carefully!!
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index") # "0,1,8,9"
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")  # 1000
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")  # 45000
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples") # 0.4
    parser.add_argument("--supervised_dataset", action="store_true", default=False, help="training with supervised dataset")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=256, type=int, help="mini-batch size (default: 256)") # 256
    # Configuration of optimization
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--cosine", action="store_true", default=True, help="use cosine lr schedule")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--wd", "--weight-decay", default=1e-3, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--epochs", type=int, default=1300, help="number of total epochs to run")
    # Configuration of saving
    parser.add_argument("--exp-dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    # Configuration of GPU utilization
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    # Checkpoint path
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none).")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "epoch_{ep}_supervised_{supervised}_seed_{seed}".format(ep=args.epochs, supervised=str(args.supervised_dataset), seed=args.seed)
    args = pre_setting(args, model_name="CE", model_path=model_path)
    if args.dataset == "stl10" and args.supervised_dataset:
        raise Exception("STL10 has not supervised label!")
    #############################################

    training_loader, testing_loader = get_ce_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    model = ResNet18(dataset_name=args.dataset).cuda()
    loss_function = CELoss(balance=True)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epochs = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epochs = checkpoint["epoch"] + 1

    print("\n==> Start Training...\n")
    best_acc = 0
    for epoch in range(start_epochs, args.epochs):
        is_best = False

        adjust_learning_rate(args, optimizer, epoch)

        train_CE(args=args, train_loader=training_loader, model=model, loss_fn=loss_function, optimizer=optimizer, epoch=epoch)

        testing_metrics = validate_CE(args=args, epoch=epoch, model=model, test_loader=testing_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint_last.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )


if __name__ == "__main__":
    args = Argparse()
    main(args)
