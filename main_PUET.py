import argparse

import numpy as np

from toolbox import pre_setting, str2lit
from utils_algorithm.one_epoch_PUET import train_PUET, validate_PUET
from utils_algorithm.PUET import PUExtraTrees
from utils_data.get_puet_dataloader import get_puet_dataloader

np.set_printoptions(suppress=True, precision=1)


def Argparse():
    parser = argparse.ArgumentParser(description="PU learning method: PUET")
    # Configuration of PU data
    parser.add_argument("--data_root", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--pos_label", type=int, default=0, help="positive label")
    parser.add_argument("--positive_class_index", type=str2lit, default="0,1,8,9", help="positive class index")
    parser.add_argument("--positive_size", type=int, default=1000, help="the number of positive training samples")
    parser.add_argument("--unlabeled_size", type=int, default=40000, help="the number of unlabeled training samples")
    parser.add_argument("--true_class_prior", type=float, default=0.4, help="proportion of positive data in unlabeled samples")
    # Configuration of saving
    parser.add_argument("--exp_dir", default="logging", type=str, help="experiment directory for saving checkpoints and logs")
    # Configuration of random seed
    parser.add_argument("--seed", type=int, default=52571314, help="random seed")
    # Configuration of PUET
    parser.add_argument("--class_prior", type=float, default=0.4, help="class prior in loss function")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--risk_estimator", type=str, default="nnPU")
    parser.add_argument("--loss", type=str, default="quadratic")
    parser.add_argument("--n_jobs", type=int, default=10)
    return parser.parse_args()


def main(args):
    #############################################
    args.gpu = 0
    model_path = "risk_estimator_{risk_estimator}_class_prior_{class_prior}_seed_{seed}".format(
        risk_estimator=args.risk_estimator, class_prior=str(args.class_prior), seed=args.seed
    )
    args = pre_setting(args, model_name="PUET", model_path=model_path)
    #############################################

    training_data, training_labels, training_true_labels, testing_data, testing_labels = get_puet_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_root,
        positive_class_index=args.positive_class_index,
        positive_num=args.positive_size,
        unlabeled_num=args.unlabeled_size,
        true_class_prior=args.true_class_prior,
        pos_label=args.pos_label,
    )

    model = PUExtraTrees(
        n_estimators=args.n_estimators,
        risk_estimator=args.risk_estimator,
        loss=args.loss,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        max_candidates=1,
        n_jobs=args.n_jobs,
    )

    print("\n==> Start Training...\n")
    train_PUET(args=args, model=model, training_data=training_data, training_labels=training_labels)

    testing_metrics = validate_PUET(args=args, model=model, testing_data=testing_data, testing_labels=testing_labels)


if __name__ == "__main__":
    args = Argparse()
    main(args)
