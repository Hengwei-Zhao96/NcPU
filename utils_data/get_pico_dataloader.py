import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetPiCO(Dataset):
    def __init__(self, images, labels, true_labels, weak_transforms, strong_transforms):
        self.images = images
        self.labels = labels
        self.true_labels = true_labels
        self.weak_transform = weak_transforms
        self.strong_transform = strong_transforms

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = torch.tensor(self.labels[index]).long()
        each_true_label = torch.tensor(self.true_labels[index]).long()

        return index, each_image_w, each_image_s, each_label, each_true_label


class TestingDatasetPiCO(Dataset):
    def __init__(self, images, true_labels, testing_transforms):
        self.images = images
        self.true_labels = true_labels
        self.testing_transforms = testing_transforms

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        testing_img = self.testing_transforms(self.images[index])
        testing_label = torch.tensor(self.true_labels[index]).long()

        return index, testing_img, testing_label


def get_pico_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, true_class_prior, batch_size, pos_label):
    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)
    temp_training_data = all_dataset["all_labeled_training_data"]
    temp_training_label = all_dataset["all_labeled_training_label"]
    test_data = all_dataset["all_testing_data"]
    testing_label = all_dataset["all_testing_label"]

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num,unlabeled_num=unlabeled_num,pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, true_class_prior=true_class_prior, pos_label=pos_label)
    
    training_p_idxs = idxs_set["training_p_idxs"]
    training_u_idxs = idxs_set["training_u_idxs"]

    training_idxs = np.concatenate((training_p_idxs, training_u_idxs))
    # Get training data and training true labels
    training_data = temp_training_data[training_idxs]
    training_true_labels = temp_training_label[training_idxs]
    # Get training labels
    training_p_labels = np.stack((np.ones(len(training_p_idxs)), np.zeros(len(training_p_idxs))), axis=1)
    training_u_labels = np.ones((len(training_u_idxs), 2))
    training_labels = np.concatenate((training_p_labels, training_u_labels), axis=0)
    # Get training confidence
    training_p_confidence = np.stack((np.ones(len(training_p_idxs)), np.zeros(len(training_p_idxs))), axis=1)
    training_u_confidence = np.stack((np.zeros(len(training_u_idxs)), np.ones(len(training_u_idxs))), axis=1)
    training_confidence = np.concatenate((training_p_confidence, training_u_confidence), axis=0)
    # Get the flag of positive and unlabeled data
    training_p_flag = np.concatenate((np.ones_like(training_p_idxs), np.zeros_like(training_u_idxs)), axis=0)
    training_u_flag = np.concatenate((np.zeros_like(training_p_idxs), np.ones_like(training_u_idxs)), axis=0)

    weak_transforms, strong_transforms, testing_transforms = get_transforms(dataset_name=dataset_name)

    training_dataset = TrainingDatasetPiCO(
        images=training_data, labels=training_labels, true_labels=training_true_labels, weak_transforms=weak_transforms, strong_transforms=strong_transforms
    )
    training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=(training_sampler is None), drop_last=True, num_workers=4, pin_memory=True, sampler=training_sampler
    )

    testing_dataset = TestingDatasetPiCO(images=test_data, true_labels=testing_label, testing_transforms=testing_transforms)
    testing_loader = DataLoader(
        testing_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=torch.utils.data.distributed.DistributedSampler(testing_dataset, shuffle=False)
    )

    return training_loader, torch.tensor(training_confidence), torch.tensor(training_p_flag), torch.tensor(training_u_flag), training_sampler, testing_loader
