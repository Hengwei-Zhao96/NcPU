import argparse
import logging
import math

import numpy as np
import torch


from toolbox import pre_setting, save_checkpoint_vPU, metric_prin, str2lit
from utils_algorithm.vPU import vPU
from utils_algorithm.one_epoch_vPU import train_vPU, cal_val_var, validate_vPU
from utils_data.get_vpu_dataloader import get_vpu_dataloader

from utils_model.resnet import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: vPU")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--val_unlabeled_size", type=int, default=4000, help="the number of unlabeled validation samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=500, type=int, help="mini-batch size (default: 256)")
    # Configuration of optimization
    parser.add_argument("--lr", type=float, default=3e-5, help="initial learning rate")#
    parser.add_argument("--epochs", type=int, default=50, help="number of total epochs to run")
    # Configuration of vPU
    parser.add_argument("--val_iterations", type=int, default=30)
    parser.add_argument("--mix_alpha", type=float, default=0.3, help="parameter in Mixup")
    parser.add_argument("--lam", type=float, default=0.03, help="weight of the regularizer")#
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "epoch_{ep}_val_iter_{val_iter}_mix_alpha_{mix_alpha}_lam_{lam}_seed_{seed}".format(
        ep=args.epochs, val_iter=args.val_iterations, mix_alpha=args.mix_alpha, lam=args.lam, seed=args.seed
    )
    args = pre_setting(args, model_name="vPU", model_path=model_path)
    #############################################

    training_p_loader, training_u_loader, validate_p_loader, validate_u_loader, testing_loader = get_vpu_dataloader(
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

    model = vPU(base_encoder=ResNet18(dataset_name=args.dataset)).cuda()
    lr_phi = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1  # highest test accuracy on test set

    print("\n==> Start Training...\n")
    for epoch in range(args.epochs):
        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        train_vPU(args=args, train_p_loader=training_p_loader, train_u_loader=training_u_loader, model=model, optimizer=optimizer, epoch=epoch)

        # calculate variational loss of the validation set consisting of PU data
        val_var = cal_val_var(model=model, val_p_loader=validate_p_loader, val_u_loader=validate_u_loader)

        testing_metrics = validate_vPU(args, model, training_p_loader, training_u_loader, testing_loader, epoch)

        # assessing performance of the current model and decide whether to save it
        is_val_var_lowest = val_var < lowest_val_var
        is_test_acc_highest = testing_metrics["OA"] > highest_test_acc
        lowest_val_var = min(lowest_val_var, val_var)
        highest_test_acc = max(highest_test_acc, testing_metrics["OA"])
        if is_val_var_lowest:
            test_metrics_of_best_val = testing_metrics
            epoch_of_best_val = epoch + 1

        save_checkpoint_vPU(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_lowest_on_val=is_val_var_lowest,
            is_highest_on_test=is_test_acc_highest,
            filepath=args.exp_dir,
        )

    best_val_log = "Early stopping at {:}th epoch ".format(epoch_of_best_val) + metric_prin(test_metrics_of_best_val)
    logging.info(best_val_log)
    print(best_val_log)


if __name__ == "__main__":
    args = Argparse()
    main(args)
