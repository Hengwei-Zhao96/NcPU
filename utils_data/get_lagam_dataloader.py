import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetLaGAM(Dataset):
    def __init__(self, images, labels, true_labels, weak_transform, strong_transform):
        self.images = images
        self.pu_labels = labels
        self.true_labels = true_labels
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def update_targets(self, new_labels, idxes):
        self.pu_labels[idxes] = new_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image_w = self.weak_transform(self.images[index])
        image_s = self.strong_transform(self.images[index])
        label = torch.tensor(self.pu_labels[index])
        true_label = torch.tensor(self.true_labels[index])

        return image_w, image_s, label, true_label, index


class EvaluateDatasetLaGAM(Dataset):
    def __init__(self, images, labels, true_labels, testing_transform):
        self.images = images
        self.pu_labels = labels
        self.true_labels = true_labels
        self.testing_transform = testing_transform

    def update_targets(self, new_labels, idxes):
        self.pu_labels[idxes] = new_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        label = torch.tensor(self.pu_labels[index])
        image = self.testing_transform(self.images[index])
        true_label = torch.tensor(self.true_labels[index])
        return image, image, label, true_label, index


class TestingDatasetLaGAM(Dataset):
    def __init__(self, images, true_labels, testing_transform):
        self.images = images
        self.true_labels = true_labels
        self.testing_transform = testing_transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image = self.testing_transform(self.images[index])
        true_label = torch.tensor(self.true_labels[index])
        return image, true_label


def get_lagam_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, n_val_num, true_class_prior, batch_size, pos_label):

    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)
    all_training_data = all_dataset["all_labeled_training_data"]
    all_training_labels = all_dataset["all_labeled_training_label"]
    all_testing_data = all_dataset["all_testing_data"]
    all_testing_label = all_dataset["all_testing_label"]

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, n_val_num=n_val_num, pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, n_val_num=n_val_num, true_class_prior=true_class_prior, pos_label=pos_label)

    training_p_idxs = idxs_set["training_p_idxs"]
    training_u_idxs = idxs_set["training_u_idxs"]
    validate_p_idxs = idxs_set["validate_p_idxs"]
    validate_uORn_dixs = idxs_set["validate_uORn_idxs"]

    training_idxs = np.concatenate((training_p_idxs, training_u_idxs))
    training_labels = np.concatenate((np.ones(len(training_p_idxs)), np.zeros(len(training_u_idxs))), axis=0)

    valid_idxs = np.concatenate((validate_p_idxs, validate_uORn_dixs))

    weak_transforms, strong_transforms, testing_transforms = get_transforms(dataset_name=dataset_name)

    train_dataset = TrainingDatasetLaGAM(
        images=all_training_data[training_idxs],
        labels=training_labels,
        true_labels=all_training_labels[training_idxs],
        weak_transform=weak_transforms,
        strong_transform=strong_transforms,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    valid_dataset = TestingDatasetLaGAM(images=all_training_data[valid_idxs], true_labels=all_training_labels[valid_idxs], testing_transform=testing_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    eval_dataset = EvaluateDatasetLaGAM(
        images=all_training_data[training_idxs], labels=training_labels, true_labels=all_training_labels[training_idxs], testing_transform=testing_transforms
    )
    eval_loader = DataLoader(eval_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    test_dataset = TestingDatasetLaGAM(images=all_testing_data, true_labels=all_testing_label, testing_transform=testing_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return train_loader, valid_loader, eval_loader, test_loader
