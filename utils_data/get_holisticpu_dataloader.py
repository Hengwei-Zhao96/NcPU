import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetHolisticPU(Dataset):
    def __init__(self, images, labels, true_labels, weak_transforms, strong_transforms):
        self.images = images
        self.labels = labels
        self.true_labels = true_labels
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image_w = self.weak_transforms(self.images[index])
        each_image_s = self.strong_transforms(self.images[index])
        each_label = torch.tensor(self.labels[index]).long()
        each_true_label = torch.tensor(self.true_labels[index]).long()

        return index, each_image_w, each_image_s, each_label, each_true_label


class TestingDatasetHolisticPU(Dataset):
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


def get_holistic_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, true_class_prior, batch_size, pos_label):
    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)
    all_training_data = all_dataset["all_labeled_training_data"]
    all_training_labels = all_dataset["all_labeled_training_label"]
    all_testing_data = all_dataset["all_testing_data"]
    all_testing_label = all_dataset["all_testing_label"]

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num,unlabeled_num=unlabeled_num,pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, true_class_prior=true_class_prior, pos_label=pos_label)

    training_p_idxs = idxs_set["training_p_idxs"]
    training_u_idxs = idxs_set["training_u_idxs"]

    # Get training positive data and labels
    training_positive_data = all_training_data[training_p_idxs]
    training_positive_labels = np.zeros(len(training_p_idxs))
    training_positive_true_labels = all_training_labels[training_p_idxs]
    # Get training unlabeled data and labels
    training_unlabeled_data = all_training_data[training_u_idxs]
    training_unlabeled_labels = np.ones(len(training_u_idxs))
    training_unlabeled_true_labels = all_training_labels[training_u_idxs]

    weak_transforms, strong_transforms, testing_transforms = get_transforms(dataset_name=dataset_name)

    training_positive_dataset = TrainingDatasetHolisticPU(
        images=training_positive_data,
        labels=training_positive_labels,
        true_labels=training_positive_true_labels,
        weak_transforms=weak_transforms,
        strong_transforms=strong_transforms,
    )
    training_positive_loader = DataLoader(training_positive_dataset, batch_size=batch_size, sampler=RandomSampler(training_positive_dataset), num_workers=4, drop_last=True)

    training_unlabeled_dataset = TrainingDatasetHolisticPU(
        images=training_unlabeled_data,
        labels=training_unlabeled_labels,
        true_labels=training_unlabeled_true_labels,
        weak_transforms=weak_transforms,
        strong_transforms=strong_transforms,
    )
    training_unlabeled_loader = DataLoader(training_unlabeled_dataset, batch_size=batch_size, sampler=SequentialSampler(training_unlabeled_dataset), num_workers=4, drop_last=True)

    testing_dataset = TestingDatasetHolisticPU(images=all_testing_data, true_labels=all_testing_label, transforms=testing_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, sampler=RandomSampler(testing_dataset), num_workers=4)

    return training_positive_loader, training_unlabeled_loader, testing_loader
