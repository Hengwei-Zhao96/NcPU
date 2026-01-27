import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter as tb_logger

from toolbox import adjust_learning_rate, str2lit
from utils_algorithm.PiCO import PiCO
from utils_algorithm.one_epoch_PiCO import train_PiCO, validate_PiCO
from utils_data.get_pico_dataloader import get_pico_dataloader
from utils_model.PiCO import SupConResNet

from utils_loss_function.PiCO import ClsLoss, SupConLoss

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: PiCO")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label") # carefully!!
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of dataloader
    parser.add_argument(
        "-b",
        "--batch_size",
        default=256,
        type=int,
        help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument("-j", "--workers", default=32, type=int, help="number of data loading workers (default: 32)")
    # Configuration of optimization
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--lr_decay_epochs", type=list, default=[700, 800, 900], help="where to decay lr")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate")
    parser.add_argument("--cosine", action="store_true", default=True, help="use cosine lr schedule")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float, metavar="W", help="weight decay (default: 1e-3)", dest="weight_decay")
    parser.add_argument("--epochs", type=int, default=1300, help="number of total epochs to run") # 800
    # Configuration for restarts
    parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    # Configuration of CoPU
    parser.add_argument("--low_dim", default=128, type=int, help="embedding dimension")
    parser.add_argument("--moco_queue", default=8192, type=int, help="queue size; number of negative samples")
    parser.add_argument("--moco_m", default=0.999, type=float, help="momentum for updating momentum encoder")
    parser.add_argument("--proto_m", default=0.99, type=float, help="momentum for computing the momving average of prototypes")
    parser.add_argument("--loss_weight", default=0.5, type=float, help="contrastive loss weight")  # 0.5
    parser.add_argument("--conf_ema_range", default=[0.99, 0.95], type=list, help="pseudo target updating coefficient (phi)")  # [0.99, 0.95]
    parser.add_argument("--prot_start", default=1, type=int, help="Start Prototype Updating")  # 1
    # Configuration of DDP, please do not modify it
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist_url", default="tcp://localhost:10001", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        default=True,
        help="Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training",
    )
    return parser.parse_args()


def main(args):

    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    model_path = "lr_{lr}_ep_{ep}_lw_{lw}_pm_{pm}_sd_{seed}".format(lr=args.lr, ep=args.epochs, lw=args.loss_weight, pm=args.proto_m, seed=args.seed)
    args.exp_dir = os.path.join(args.exp_dir, args.dataset, "PiCO", str(args.positive_class_index), model_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    with open(os.path.join(args.exp_dir, "result.log"), "a+") as f:
        f.write(str(args) + "\n")

    ngpus_per_node = torch.cuda.device_count()
    # ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model")
    model = PiCO(args, SupConResNet)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    training_loader, training_confidence, training_p_flag, training_u_flag, training_sampler, testing_loader = get_pico_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        batch_size=args.batch_size,
        pos_label=args.pos_label,
    )

    # protopy is updated in the PiCO model during forward
    # training confidence is updated in the classification loss function
    loss_cls = ClsLoss(confidence=training_confidence, positive_flag=training_p_flag, unlabeled_flag=training_u_flag)
    loss_cont = SupConLoss()

    if args.gpu == 0:
        logger = tb_logger(log_dir=os.path.join(args.exp_dir, "tensorboard"), flush_secs=2)

    else:
        logger = None

    print("\nStart Training\n")
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        start_upd_prot = epoch >= args.prot_start
        if args.distributed:
            training_sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)
        log_str_list = train_PiCO(
            args=args,
            train_loader=training_loader,
            model=model,
            loss_cls_fn=loss_cls,
            loss_cont_fn=loss_cont,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=logger,
            start_upd_prot=start_upd_prot,
        )

        loss_cls.set_conf_ema_m(epoch, args)

        testing_metrics, testing_prin = validate_PiCO(args=args, epoch=epoch, model=model, test_loader=testing_loader, tb_logger=logger)
        log_str_list.append(testing_prin)

        with open(os.path.join(args.exp_dir, "result.log"), "a+") as f:
            for log in log_str_list:
                f.write(log + "\n")
        if testing_metrics["OA"].item() > best_acc:
            best_acc = testing_metrics["OA"].item()
            is_best = True

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename="{}/checkpoint.pth.tar".format(args.exp_dir),
                best_file_name="{}/checkpoint_best.pth.tar".format(args.exp_dir),
            )


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_file_name="model_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


if __name__ == "__main__":
    args = Argparse()
    main(args)
