import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetvPU(Dataset):
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


class TestingDatasetvPU(Dataset):
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


def get_vpu_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, u_val_num, true_class_prior, batch_size, pos_label):

    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)
    all_training_data = all_dataset["all_labeled_training_data"]
    all_training_labels = all_dataset["all_labeled_training_label"]
    all_testing_data = all_dataset["all_testing_data"]
    all_testing_label = all_dataset["all_testing_label"]

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num,unlabeled_num=unlabeled_num,u_val_num=u_val_num,pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num,u_val_num=u_val_num, true_class_prior=true_class_prior, pos_label=pos_label)
    
    training_p_idxs = idxs_set["training_p_idxs"]
    training_u_idxs = idxs_set["training_u_idxs"]
    validate_p_idxs = idxs_set["validate_p_idxs"]
    validate_uORn_dixs = idxs_set["validate_uORn_idxs"]

    training_transforms, _, testing_transforms = get_transforms(dataset_name=dataset_name)

    training_p_dataset = TrainingDatasetvPU(
        images=all_training_data[training_p_idxs], labels=np.zeros(len(training_p_idxs)), true_labels=all_training_labels[training_p_idxs], transforms=training_transforms
    )
    training_p_loader = DataLoader(training_p_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    training_u_dataset = TrainingDatasetvPU(
        images=all_training_data[training_u_idxs], labels=np.ones(len(training_u_idxs)), true_labels=all_training_labels[training_u_idxs], transforms=training_transforms
    )
    training_u_loader = DataLoader(training_u_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    validate_p_dataset = TrainingDatasetvPU(
        all_training_data[validate_p_idxs], labels=np.zeros(len(validate_p_idxs)), true_labels=all_training_labels[validate_p_idxs], transforms=testing_transforms
    )
    validate_p_loader = DataLoader(validate_p_dataset, batch_size=batch_size, shuffle=False)

    validate_u_dataset = TrainingDatasetvPU(
        all_training_data[validate_uORn_dixs], labels=np.ones(len(validate_uORn_dixs)), true_labels=all_training_labels[validate_uORn_dixs], transforms=testing_transforms
    )
    validate_u_loader = DataLoader(validate_u_dataset, batch_size=batch_size, shuffle=False)

    testing_dataset = TestingDatasetvPU(images=all_testing_data, true_labels=all_testing_label, transforms=testing_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size)

    return training_p_loader, training_u_loader, validate_p_loader, validate_u_loader, testing_loader
