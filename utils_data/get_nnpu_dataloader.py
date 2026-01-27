import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetnnPU(Dataset):
    def __init__(self, images, labels, true_labels, transforms):
        self.images = images
        self.labels = labels
        self.true_labels = true_labels
        self.transform = transforms

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.transform(self.images[index])
        each_label = torch.tensor(self.labels[index]).long()
        each_true_label = torch.tensor(self.true_labels[index]).long()

        return index, each_image, each_label, each_true_label


class TestingDatasetnnPU(Dataset):
    def __init__(self, images, true_labels, transforms):
        self.images = images
        self.true_labels = true_labels
        self.transform = transforms

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.transform(self.images[index])
        each_true_label = torch.tensor(self.true_labels[index]).long()

        return index, each_image, each_true_label


def get_nnpu_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, true_class_prior, batch_size, pos_label):
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

    training_transforms, _, testing_transforms = get_transforms(dataset_name=dataset_name)

    training_dataset = TrainingDatasetnnPU(images=training_data, labels=training_labels, true_labels=training_true_labels, transforms=training_transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    testing_dataset = TestingDatasetnnPU(images=all_dataset["all_testing_data"], true_labels=all_dataset["all_testing_label"], transforms=testing_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return training_loader, testing_loader
