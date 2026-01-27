import argparse

import numpy as np
import torch

from toolbox import pre_setting, adjust_learning_rate, save_checkpoint, str2lit
from utils_algorithm.LaGAM import create_model, run_kmeans
from utils_data.get_lagam_dataloader import get_lagam_dataloader
from utils_loss_function.LaGAM import BCELoss, ContLoss
from utils_algorithm.one_epoch_LaGAM import train_LaGAM, validate_LaGAM, compute_features

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: LaGAM")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=1, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--val_negative_size", type=int, default=500, help="the number of validation negative samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size (default: 256)")
    # Configuration of optimization
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--lr_decay_epochs", type=list, default=[250, 300, 350], help="where to decay lr")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate")
    parser.add_argument("--cosine", action="store_true", default=False, help="use cosine lr schedule")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--epochs", type=int, default=400, help="number of total epochs to run")
    # Configuration for restarts
    parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    # Configuration of LaGAM
    parser.add_argument("--mix_weight", default=1.0, type=float, help="mixup loss weight")
    parser.add_argument("--rho_range", default=[0.95, 0.8], type=list, help="momentum updating parameter")
    parser.add_argument("--warmup_epoch", default=20, type=int, help="epoch number of warm up")
    parser.add_argument("--num_cluster", default=5, type=int, help="number of clusters")
    parser.add_argument("--temperature", default=0.07, type=float, help="mixup loss weight")
    parser.add_argument("--cont_cutoff", action="store_true", default=True, help="whether cut off by classifier")
    parser.add_argument("--knn_aug", action="store_true", default=True, help="whether using kNN for CL")
    parser.add_argument("--num_neighbors", default=10, type=int, help="number of neighbors")
    parser.add_argument("--identifier", default="classifier", type=str, help="identifier for meta layers, e.g. classifier")
    parser.add_argument("--contrastive_clustering", default=1, type=int, help="whether using contrastive clustering")
    # Configuration of GPU
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    return parser.parse_args()


def main(args):
    #############################################
    [args.rho_start, args.rho_end] = args.rho_range
    model_path = "lr_{lr}_epoch_{ep}_warm_up_{warm_up}_cont_cutoff_{cont_cutoff}_knn{knn}{k}_seed_{seed}".format(
        lr=args.lr, ep=args.epochs, warm_up=args.warmup_epoch, cont_cutoff=args.cont_cutoff, knn=args.knn_aug, k=args.num_neighbors, seed=args.seed
    )
    args = pre_setting(args, model_name="LaGAM", model_path=model_path)
    #############################################

    train_loader, valid_loader, eval_loader, test_loader = get_lagam_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        n_val_num=args.val_negative_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    model = create_model(num_class=1, dataset_name=args.dataset).cuda()
    optimizer = torch.optim.SGD(model.params(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    bce_loss = BCELoss()
    contrastive_loss = ContLoss(
        temperature=args.temperature, cont_cutoff=args.cont_cutoff, knn_aug=args.knn_aug, num_neighbors=args.num_neighbors, contrastive_clustering=args.contrastive_clustering
    )

    best_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        if epoch < args.warmup_epoch:
            train_LaGAM(
                args=args, train_loader=train_loader, valid_loader=valid_loader, model=model, optimizer=optimizer, bce_loss=bce_loss, contrastive_loss=contrastive_loss, epoch=epoch
            )
        else:
            features = compute_features(model=model, eval_loader=eval_loader)
            cluster_result = run_kmeans(features, args)
            train_LaGAM(
                args=args,
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                optimizer=optimizer,
                bce_loss=bce_loss,
                contrastive_loss=contrastive_loss,
                epoch=epoch,
                cluster_result=cluster_result,
            )

        testing_metrics = validate_LaGAM(args=args, epoch=epoch, model=model, test_loader=test_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )


if __name__ == "__main__":
    args = Argparse()
    main(args)
