import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

from .utils_get_datasets import get_dataset, train_val_split, train_val_split_with_unlabeled
from .utils_get_transforms import get_transforms


class TrainingDatasetDistPU(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transform = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        each_image = self.transform(self.images[index])
        each_label = torch.tensor(self.labels[index]).long()

        return index, each_image, each_label


class TestingDatasetDistPU(Dataset):
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


class MixupDataset:
    def __init__(self) -> None:
        self.psudo_labels = None
        pass

    def update_psudos(self, data_loader, model):
        model.eval()
        predicted_scores = []
        indexes = []

        with torch.no_grad():
            for _, (index, Xs, Ys) in enumerate(data_loader):
                Xs = Xs.cuda()
                Ys = Ys.cuda()
                outputs = model(Xs).squeeze()

                outputs = torch.sigmoid(outputs)

                predicted_scores.append(outputs)
                indexes.append(index.squeeze())

        self.psudo_labels = torch.cat(predicted_scores)
        self.indexes = torch.cat(indexes)


def get_distpu_dataloader(dataset_name, data_path, positive_class_index, positive_num, unlabeled_num, true_class_prior, batch_size, pos_label):
    all_dataset = get_dataset(dataset_name=dataset_name, data_path=data_path, positive_class_index=positive_class_index, pos_label=pos_label)

    if dataset_name == "stl10":
        assert true_class_prior==0, "Class prior must be 0!"
        idxs_set = train_val_split_with_unlabeled(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num,unlabeled_num=unlabeled_num,pos_label=pos_label)
    else:
        idxs_set = train_val_split(labels=all_dataset["all_labeled_training_label"], positive_num=positive_num, unlabeled_num=unlabeled_num, true_class_prior=true_class_prior, pos_label=pos_label)

    training_idxs = np.concatenate((idxs_set["training_p_idxs"], idxs_set["training_u_idxs"]))
    training_data = all_dataset["all_labeled_training_data"][training_idxs]
    training_labels = np.concatenate((np.ones(len(idxs_set["training_p_idxs"])), np.zeros(len(idxs_set["training_u_idxs"]))), axis=0)

    training_transforms, _, testing_transforms = get_transforms(dataset_name=dataset_name)

    training_dataset = TrainingDatasetDistPU(images=training_data, labels=training_labels, transforms=training_transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    testing_dataset = TestingDatasetDistPU(images=all_dataset["all_testing_data"], true_labels=all_dataset["all_testing_label"], transforms=testing_transforms)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    mixup_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)

    return training_loader, testing_loader, mixup_loader
