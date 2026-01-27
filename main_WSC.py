import argparse

import numpy as np
import torch
import torch.nn as nn


from toolbox import save_checkpoint, pre_setting, str2lit
from utils_algorithm.WSC import NoiseMatrixLayer, param_groups_weight_decay
from utils_algorithm.one_epoch_WSC import train_WSC, validate_WSC
from utils_data.get_wsc_dataloader import get_wsc_dataloader
from utils_loss_function.WSC import SupervisedNoisyLoss, WeakSpectralLoss
from utils_model.WSC import ResNet18

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: WSC")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=256, type=int, help="mini-batch size (default: 256)")
    
    # Configuration of optimization
    parser.add_argument("--lr", default=0.02, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--wd", "--weight-decay", default=0.001, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--epochs", type=int, default=250, help="number of total epochs to run")
    # Configuration of WSC model
    parser.add_argument("--average_entropy_loss", action='store_true', default=False, help="use average entropy loss")
    parser.add_argument("--noise_matrix_scale", type=float, default=0.5, help="scale for noise matrix, should be big when number of class is large")
    parser.add_argument("--vol_lambda", type=float, default=0.0001, help="lambda for VolMinNet loss")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha for wsc loss")
    parser.add_argument("--beta", type=float, default=12, help="beta for wsc loss")
    parser.add_argument("--lam", type=float, default=1, help="lambda for wsc loss")
    parser.add_argument("--lam_consist", type=float, default=3, help="lambda for consistency loss in wsc loss")
    parser.add_argument("--balance_lam", type=float, default=0.1)
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    # Configuration of GPU utilization
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    model_path = "epoch_{ep}_seed_{seed}".format(ep=args.epochs, seed=args.seed)
    args = pre_setting(args, model_name="WSC", model_path=model_path)
    #############################################

    training_loader, testing_loader = get_wsc_dataloader(
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
    noise_model = NoiseMatrixLayer(num_classes=2, init=args.noise_matrix_scale)

    def create_projector(in_dim, out_dim):
        squential = nn.Sequential(nn.Linear(in_dim, in_dim),
                            nn.BatchNorm1d(in_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_dim, out_dim),
                            nn.BatchNorm1d(out_dim),
                            )
        return squential

    projector = create_projector(512, 256).cuda()

    sup_loss = SupervisedNoisyLoss(balance=True)
    weak_spec_loss = WeakSpectralLoss(args.alpha, args.beta, args)
    
    per_param_args = param_groups_weight_decay(model, args.weight_decay, no_weight_decay_list={})
    optimizer = torch.optim.SGD(per_param_args, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    noise_matrix_optimizer = torch.optim.SGD(
        noise_model.parameters(),
        lr=args.lr,
        momentum=0,
        weight_decay=0
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(args.epochs * len(training_loader)), eta_min=2e-4)

    print("\n==> Start Training...\n")
    best_acc = 0
    for epoch in range(args.epochs):

        train_WSC(args=args, train_loader=training_loader, model=model, noise_model=noise_model, projector=projector, optimizer=optimizer, noise_matrix_optimizer=noise_matrix_optimizer,  sup_loss=sup_loss, weak_spec_loss=weak_spec_loss, scheduler=scheduler, epoch=epoch)

        testing_metrics = validate_WSC(args=args, epoch=epoch, model=model, test_loader=testing_loader)

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
