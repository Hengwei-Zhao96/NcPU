import os
import logging
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

def str2lit(s):
    return [int(x) for x in s.split(',')]

def pre_setting(args, model_name, model_path):
    # GPU
    torch.cuda.set_device(args.gpu)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # random seed
    if args.seed is not None:
        print("You have chosen to seed training.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # save path
    args.exp_dir = os.path.join(args.exp_dir, args.dataset, model_name, model_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # logging
    logging.basicConfig(filename=os.path.join(args.exp_dir, "logging.log"), filemode="w", format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logging.info(args)
    logging.info("---------------------Pre Setting Completed---------------------")
    args.tb_logger = SummaryWriter(log_dir=os.path.join(args.exp_dir, "tensorboard"), flush_secs=2)

    return args
