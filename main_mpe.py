import argparse
import numpy as np
import logging

from toolbox.class_prior_estimation import KM_estimate

from utils_data.get_vpu_dataloader import get_vpu_dataloader
from toolbox import pre_setting, str2lit

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU Learning: MPE")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name") # "cifar10"
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index") # "0,1,8,9"
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")  # 1000
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")  # 40000
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples") # 0.4
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument('--seed', type=int, default=52571314, help='random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, default='KMPE', help='MPE Methods')
    return parser.parse_args()


def main(args):
    model_path=args.method
    args = pre_setting(args, model_name="MPE", model_path=model_path)
    
    training_p_loader, training_u_loader, _, _, _ = get_vpu_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        u_val_num=40,
        true_class_prior=args.true_class_prior,
        batch_size=256,
        pos_label=args.pos_label,
    )

    X = training_p_loader.dataset.images.reshape(len(training_p_loader.dataset.images), -1)
    Mix = training_u_loader.dataset.images.reshape(len(training_u_loader.dataset.images), -1)

    # The size of data be set to 2000(P) and 2000(U)
    if args.method == "KMPE":
        kmpe1, kmpe2 = KM_estimate(p_data=X, u_data=Mix)
        print("KM1:"+str(kmpe1))
        print("KM2:"+str(kmpe2))
        logging.info("KM1:"+str(kmpe1))
        logging.info("KM2:"+str(kmpe2))
    else:
        raise NotImplementedError("Wrong KMPE arguments.")


if __name__ == '__main__':
    args = Argparse()
    main(args)
