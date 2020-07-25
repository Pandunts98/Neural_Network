import numpy as np
import json

"""
Define all metrics (e.g. accuracy, AUC, f1-score, etc.) here.
"""


def accuracy_score(Y_hat, Y_true):
    return np.sum(Y_hat == Y_true) / Y_true.shape[0]


def one_hot(labels):
    labels = labels.astype(np.int64)
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1.
    return y


def export_params(data):
    with open('params.json', 'w') as json_file:
        json.dump(data, json_file)


