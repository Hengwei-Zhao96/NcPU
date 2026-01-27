import logging

import numpy as np
import torch

from toolbox import pu_metric, metric_prin


def train_PUET(args, model, training_data, training_labels):
    p_idxs = np.where(training_labels == 0)[0]
    u_idxs = np.where(training_labels == 1)[0]

    p_data = training_data[p_idxs].reshape(len(p_idxs), -1)
    u_data = training_data[u_idxs].reshape(len(u_idxs), -1)

    model.fit(P=p_data, U=u_data, pi=args.class_prior)


def validate_PUET(args, model, testing_data, testing_labels):
    print("==> Evaluation...")

    testing_data = testing_data.reshape(len(testing_labels), -1)

    y_pred = np.where(model.predict(testing_data) > 0, 0, 1)

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(testing_labels)

    testing_metrics = pu_metric(y_true, y_pred, pos_label=args.pos_label)
    testing_prin = metric_prin(testing_metrics)
    logging.info(testing_prin)
    print(testing_prin)
