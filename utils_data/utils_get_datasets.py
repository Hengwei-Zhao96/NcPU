import ssl
import numpy as np
import torchvision.datasets as dsets


def binarize_labels(labels, positive_class_index, pos_label):
    """
    outputs:numpy.adarry
    """
    assert pos_label in [0, 1], "Check pos_label parameter again!"

    pn_label = []
    for l in labels:
        if l in positive_class_index:
            pn_label.append(pos_label)
        elif l == -1:
            pn_label.append(-1)
        else:
            pn_label.append(1-pos_label)
    
    return np.asarray(pn_label)

def get_dataset(dataset_name, data_path, positive_class_index, pos_label):
    """
    outputs:
    all_dataset = {
    "all_labeled_training_data": all labeled training data (numpy.ndarry),
    "all_labeled_training_label": all labeled training labels (numpy.ndarry),
    "all_testing_data": all testing data (numpy.ndarry),
    "all_testing_label": all testing label (numpy.ndarry)
    }
    """

    all_dataset = dict()

    if dataset_name == "cifar10":
        # positive_class_index: "0,1,8,9"
        # positive_size: 1000
        # unlabeled_size: 45000
        # true_class_prior: 0.4
        # batch_size: 256
        # lr: 0.001
        # ent_loss_weight: 5
        all_training_dataset = dsets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        all_testing_dataset = dsets.CIFAR10(root=data_path, train=False, transform=None)

        all_labeled_training_data = np.array(all_training_dataset.data)
        all_labeled_training_label = binarize_labels(np.array(all_training_dataset.targets), positive_class_index, pos_label)
        all_testing_data = np.array(all_testing_dataset.data)
        all_testing_label = binarize_labels(np.array(all_testing_dataset.targets), positive_class_index, pos_label)
    elif dataset_name == "cifar100":
        # positive_class_index: "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80"
        # positive_size: 1000
        # unlabeled_size: 45000
        # true_class_prior: 0.5
        # batch_size: 256
        # lr: 0.001
        # ent_loss_weight: 0.5
        all_training_dataset = dsets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        all_testing_dataset = dsets.CIFAR100(root=data_path, train=False, transform=None)

        all_labeled_training_data = np.array(all_training_dataset.data)
        all_labeled_training_label = binarize_labels(np.array(all_training_dataset.targets), positive_class_index, pos_label)
        all_testing_data = np.array(all_testing_dataset.data)
        all_testing_label = binarize_labels(np.array(all_testing_dataset.targets), positive_class_index, pos_label)
    elif dataset_name == "stl10":
        # positive_class_index: "0,2,3,8,9"
        # positive_size: 1000
        # unlabeled_size: 100000
        # true_class_prior: 0 (Unknown)
        # batch_size: 512
        # lr: 0.001
        # ent_loss_weight: 0.5
        all_training_dataset = dsets.STL10(root=data_path, split="train+unlabeled", download=True)
        all_testing_dataset = dsets.STL10(root=data_path, split="test")
        
        all_labeled_training_data = np.array(all_training_dataset.data).transpose(0,2,3,1)
        all_labeled_training_label = binarize_labels(np.array(all_training_dataset.labels), positive_class_index, pos_label)
        all_testing_data = np.array(all_testing_dataset.data).transpose(0,2,3,1)
        all_testing_label = binarize_labels(np.array(all_testing_dataset.labels), positive_class_index, pos_label)
    else:
        raise NotImplementedError("Wrong dataset arguments.")

    all_dataset["all_labeled_training_data"] = all_labeled_training_data
    all_dataset["all_labeled_training_label"] = all_labeled_training_label
    all_dataset["all_testing_data"] = all_testing_data
    all_dataset["all_testing_label"] = all_testing_label

    print("\n==> Preparing data...")
    print("    # test data: ", len(all_testing_label))

    return all_dataset


def train_val_split(labels, positive_num, unlabeled_num, true_class_prior, pos_label, n_val_num=0, u_val_num=0):
    """
    outputs:
    idxs_set = {
    "training_p_idxs": positive idxs for training (numpy.ndarry),
    "training_u_idxs": unlabeled idxs for training (numpy.ndarry),
    "validate_p_idxs": positive idxs for validation (numpy.ndarry),
    "validate_uORn_idxs": unlabeled or negative idxs for validation (numpy.ndarry)
    }
    """

    np.random.seed(57)
    idxs_set = dict()

    p_idxs = np.where(labels == pos_label)[0]
    n_idxs = np.where(labels != pos_label)[0]
    np.random.shuffle(p_idxs)
    np.random.shuffle(n_idxs)

    unlabeled_p_num = int(unlabeled_num * true_class_prior)
    unlabeled_n_num = int(unlabeled_num * (1 - true_class_prior))

    if u_val_num != 0 and n_val_num == 0:
        val_p_num = int(positive_num / unlabeled_num * u_val_num)
        val_up_num = int(u_val_num * true_class_prior)
        val_n_num = int(u_val_num * (1 - true_class_prior))
    elif u_val_num == 0 and n_val_num != 0:
        val_p_num = int(n_val_num)
        val_up_num = int(0)
        val_n_num = n_val_num
    elif u_val_num != 0 and n_val_num != 0:
        raise Exception("n_val_num and u_val_num can not be used simultaneously.")
    else:
        val_p_num = 0
        val_up_num = 0
        val_n_num = 0

    # check the number of positive and negative data
    assert positive_num + unlabeled_p_num + val_p_num + val_up_num <= len(p_idxs), "Check positive sizes again!"
    assert unlabeled_n_num + val_n_num <= len(n_idxs), "Check negative sizes again!"

    training_p_idxs = p_idxs[:positive_num]
    training_u_idxs = np.concatenate((p_idxs[positive_num : positive_num + unlabeled_p_num], n_idxs[:unlabeled_n_num]))
    np.random.shuffle(training_u_idxs)
    idxs_set["training_p_idxs"] = training_p_idxs
    idxs_set["training_u_idxs"] = training_u_idxs

    print("    # training positive data: ", len(training_p_idxs))
    print("    # training unlabeled data: ", len(training_u_idxs))

    if u_val_num != 0 or n_val_num != 0:
        validate_p_idxs = p_idxs[positive_num + unlabeled_p_num : positive_num + unlabeled_p_num + val_p_num]
        if n_val_num == 0:
            validate_uORn_dixs = np.concatenate(
                (
                    p_idxs[positive_num + unlabeled_p_num + val_p_num : positive_num + unlabeled_p_num + val_p_num + val_up_num],
                    n_idxs[unlabeled_n_num : unlabeled_n_num + val_n_num],
                )
            )
        elif u_val_num == 0:
            validate_uORn_dixs = n_idxs[unlabeled_n_num : unlabeled_n_num + val_n_num]
        np.random.shuffle(validate_uORn_dixs)

        idxs_set["validate_p_idxs"] = validate_p_idxs
        idxs_set["validate_uORn_idxs"] = validate_uORn_dixs

        print("    # validate positive data: ", len(validate_p_idxs))
        print("    # validate uORn data: ", len(validate_uORn_dixs))

    return idxs_set

def train_val_split_with_unlabeled(labels, positive_num, unlabeled_num, pos_label, n_val_num=0, u_val_num=0):
    """
    outputs:
    idxs_set = {
    "training_p_idxs": positive idxs for training (numpy.ndarry),
    "training_u_idxs": unlabeled idxs for training (numpy.ndarry),
    "validate_p_idxs": positive idxs for validation (numpy.ndarry),
    "validate_uORn_idxs": unlabeled or negative idxs for validation (numpy.ndarry)
    }
    """

    np.random.seed(57)
    idxs_set = dict()

    p_idxs = np.where(labels == pos_label)[0]
    u_idxs = np.where(labels == -1)[0]
    n_idxs = np.where(labels == 1 - pos_label)[0]
    np.random.shuffle(p_idxs)
    np.random.shuffle(u_idxs)
    np.random.shuffle(n_idxs)

    if u_val_num != 0 and n_val_num == 0:
        val_p_num = int(positive_num / unlabeled_num * u_val_num)
        val_n_num = 0
    elif u_val_num == 0 and n_val_num != 0:
        val_p_num = int(n_val_num)
        val_n_num = n_val_num
    elif u_val_num != 0 and n_val_num != 0:
        raise Exception("n_val_num and u_val_num can not be used simultaneously.")
    else:
        val_p_num = 0
        val_n_num = 0

    # check the number of positive and negative data
    assert positive_num + val_p_num <= len(p_idxs), "Check positive sizes again!"
    assert unlabeled_num + u_val_num <= len(u_idxs), "Check unlabeled sizes again!"
    assert val_n_num <= len(n_idxs), "Check negative sizes again!"

    training_p_idxs = p_idxs[:positive_num]
    training_u_idxs = u_idxs[:unlabeled_num]

    idxs_set["training_p_idxs"] = training_p_idxs
    idxs_set["training_u_idxs"] = training_u_idxs

    print("    # training positive data: ", len(training_p_idxs))
    print("    # training unlabeled data: ", len(training_u_idxs))

    if u_val_num != 0 or n_val_num != 0:
        validate_p_idxs = p_idxs[positive_num: positive_num + val_p_num]
        if n_val_num == 0:
            validate_uORn_dixs = u_idxs[unlabeled_num:unlabeled_num + u_val_num]
        elif u_val_num == 0:
            validate_uORn_dixs = n_idxs[:val_n_num]
        np.random.shuffle(validate_uORn_dixs)

        idxs_set["validate_p_idxs"] = validate_p_idxs
        idxs_set["validate_uORn_idxs"] = validate_uORn_dixs

        print("    # validate positive data: ", len(validate_p_idxs))
        print("    # validate uORn data: ", len(validate_uORn_dixs))

    return idxs_set
