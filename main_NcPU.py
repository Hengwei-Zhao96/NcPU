import argparse
import os

import numpy as np
import torch

from toolbox import pre_setting, adjust_learning_rate, save_checkpoint, str2lit
from utils_algorithm.NoiCPU import NoiCPU
from utils_algorithm.one_epoch_NoiCPU import train_NoiCPU, validate_NoiCPU
from utils_data.get_noicpu_dataloader import get_noicpu_dataloader

from utils_loss_function.NoiCPU import ClsLoss, EntropyLoss, NoiContLoss

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="NoiCPU")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name") # "cifar10"
    parser.add_argument("--pos_label", type=int, default=0, help="positive label") # carefully!!
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index") # "0,1,8,9"
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")  # 1000
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")  # 40000
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples") # 0.4
    # Configuration of dataloader
    parser.add_argument("-b", "--batch_size", default=256, type=int, help="mini-batch size (default: 256)") # 256
    # Configuration for optimization
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--cosine", action="store_true", default=True, help="use cosine lr schedule")
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--epochs", type=int, default=1300, help="number of total epochs to run")
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    # Configuration of NoiCPU
    ## Loss function
    parser.add_argument("--warm_up", default=30, type=int, help="the number of epochs for warming up")
    parser.add_argument("--loss_type", default="ce", type=str, choices=['ce', 'tce'], help="classification loss type")
    parser.add_argument("--contloss_type", default="noisy_cont", type=str, choices=['noisy_cont', 'clean_cont', 'self_cont'], help="contrastive loss type")
    parser.add_argument('--tau', type=float, default=0.5, help='contrastive threshold (tau)')
    parser.add_argument("--cont_loss_weight", default=50, type=float, help="contrastive loss weight")
    parser.add_argument("--ent_loss_weight", default=5, type=float, help="entropy loss weight")
    parser.add_argument("--class_prior", default=None, type=float, help="class prior for SAT")
    ## Prototype param
    parser.add_argument('--model_m', type=float, default=0.99, help='moving average of the target network')
    parser.add_argument("--proto_m", type=float, default=0.99, help="momentum for computing the momving average of prototypes")
    parser.add_argument("--conf_ema_m", default=0.99, type=float, help="momentum for confidence")
    parser.add_argument("--threshold_m", default=0.99, type=float, help="momentum for threshold")
    # Configuration of GPU utilization
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    # Checkpoint path
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none).")
    return parser.parse_args()

def main(args):
    #############################################
    model_path = os.path.join(
        "epoch_{ep}_warm_up_{warm_up}".format(ep=args.epochs, warm_up=args.warm_up),
        "loss_type_{loss_type}_contloss_type_{contloss_type}_tau_{tau}_class_prior_{class_prior}".format(loss_type=args.loss_type, contloss_type=args.contloss_type, tau=args.tau, class_prior=args.class_prior),
        "cont_loss_weight_{cont_loss_weight}_ent_loss_weight_{ent_loss_weight}".format(cont_loss_weight=args.cont_loss_weight, ent_loss_weight=args.ent_loss_weight),
        "model_m_{model_m}_proto_m_{proto_m}_conf_ema_m_{conf_ema_m}_threshold_m_{threshold_m}".format(model_m=args.model_m, proto_m=args.proto_m, conf_ema_m=args.conf_ema_m, threshold_m=args.threshold_m),
        "seed_{seed}".format(seed=args.seed)
        )
    args = pre_setting(args, model_name="NoiCPU", model_path=model_path)
    #############################################

    training_loader, testing_loader, training_confidence = get_noicpu_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label)
    
    model = NoiCPU(args, label_confidence = training_confidence.clone()).cuda()
    del training_confidence

    scaler = torch.GradScaler()

    cls_loss = ClsLoss(args)
    cont_loss = NoiContLoss(contloss_type = args.contloss_type)
    ent_loss = EntropyLoss()

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
        train_NoiCPU(args=args, train_loader=training_loader,model=model,cls_loss_fn=cls_loss, cont_loss_fn=cont_loss, ent_loss_fn=ent_loss, optimizer=optimizer, scaler=scaler, epoch=epoch)

        testing_metrics = validate_NoiCPU(args=args, epoch=epoch, model=model, test_loader=testing_loader)

        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        save_checkpoint(
            state={"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            is_best=is_best,
            filename="{}/checkpoint_last.pth.tar".format(args.exp_dir),
            best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
        )

if __name__=="__main__":
    args = Argparse()
    main(args)