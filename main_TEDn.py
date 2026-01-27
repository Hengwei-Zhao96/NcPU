import argparse
import logging

import numpy as np
import torch
import torch.nn as nn


from toolbox import pre_setting, save_checkpoint, str2lit
from toolbox.class_prior_estimation import BBE_estimator
from utils_algorithm.TEDn import rank_inputs
from utils_algorithm.one_epoch_TEDn import train_TEDn_warmup, train_TEDn, validate_TEDn
from utils_data.get_tedn_dataloader import get_tedn_dataloader

from utils_model.resnet import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: TEDn")
    # Configuration of PU data
    parser.add_argument("--data-root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--val_unlabeled_size", type=int, default=4000, help="the number of unlabeled validation samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=200, type=int, help="mini-batch size (default: 256)")
    # Configuration of optimization
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of total epochs to run")
    parser.add_argument("--warm-start-epochs", type=int, default=100, help="Epochs for domain discrimination training")
    parser.add_argument("--wd", "--weight_decay", default=5e-4, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("--train_interval", type=int, default=30)
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "epoch_{ep}_warm_epochs_{warm_epochs}_seed_{seed}".format(ep=args.epochs, warm_epochs=args.warm_start_epochs, seed=args.seed)
    args = pre_setting(args, model_name="TEDn", model_path=model_path)
    #############################################

    p_trainloader, u_trainloader, p_validloader, u_validloader, testing_loader = get_tedn_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        u_val_num=args.val_unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    estimated_class_prior = None

    train_unlabeled_size = len(u_trainloader.dataset)

    model = ResNet18(dataset_name=args.dataset).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print("\n==> Start Warmup Training...\n")
    best_acc = 0
    for epoch in range(args.warm_start_epochs):
        is_best = False

        train_TEDn_warmup(args=args, epoch=epoch, net=model, p_trainloader=p_trainloader, u_trainloader=u_trainloader, optimizer=optimizer, criterion=criterion)

        testing_metrics = validate_TEDn(args=args, epoch=epoch, model=model, test_loader=testing_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )

        class_prior, _, _ = BBE_estimator(p_loader=p_validloader, u_loader=u_validloader, model=model)

        estimated_class_prior = class_prior

        class_prior_log = "Estimated Class Prior: " + str(estimated_class_prior) + "\n"
        logging.info(class_prior_log)
        print(class_prior_log)

    print("\n==> Start Training...\n")
    for epoch in range(args.epochs):
        is_best = False

        keep_samples = rank_inputs(net=model, u_trainloader=u_trainloader, alpha=estimated_class_prior, u_size=train_unlabeled_size)

        train_TEDn(args=args, epoch=epoch, net=model, p_trainloader=p_trainloader, u_trainloader=u_trainloader, optimizer=optimizer, criterion=criterion, keep_sample=keep_samples)

        testing_metrics = validate_TEDn(args=args, epoch=args.warm_start_epochs + epoch, model=model, test_loader=testing_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )

        class_prior, _, _ = BBE_estimator(p_loader=p_validloader, u_loader=u_validloader, model=model)

        estimated_class_prior = class_prior

        class_prior_log = "Estimated Class Prior: " + str(estimated_class_prior) + "\n"
        logging.info(class_prior_log)
        print(class_prior_log)


if __name__ == "__main__":
    args = Argparse()
    main(args)
