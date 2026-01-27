import numpy as np

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled


def get_puet_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, true_class_prior, pos_label):
    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num,unlabeled_num=unlabeled_num,pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, true_class_prior=true_class_prior, pos_label=pos_label)

    training_idxs = np.concatenate((idxs_set["training_p_idxs"], idxs_set["training_u_idxs"]))
    # Get training data and training true labels
    training_data = all_dataset["all_labeled_training_data"][training_idxs]
    training_true_labels = all_dataset["all_labeled_training_label"][training_idxs]
    # Get training labels
    training_labels = np.concatenate((np.zeros(len(idxs_set["training_p_idxs"])), np.ones(len(idxs_set["training_u_idxs"]))), axis=0)
    # Get testing dataset
    testing_data = all_dataset["all_testing_data"]
    testing_labels = all_dataset["all_testing_label"]

    return training_data, training_labels, training_true_labels, testing_data, testing_labels
