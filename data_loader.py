import mat73
import numpy as np
import scipy.io as sio
import torch
import random
from torch.utils.data import Dataset, DataLoader
from utils import *
def get_pairs(E_X, E_Y, neg_prop, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(E_X)):
        view0.append(E_X[i])
        view1.append(E_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])
    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    for j in range(len(E_X)):
        neg_idx = random.sample(range(len(E_Y)), neg_prop)
        for k in range(neg_prop):
            view0.append(E_X[j])
            view1.append(E_Y[neg_idx[k]])
            labels.append(0)
            class_labels0.append(train_label[j])
            class_labels1.append(train_label[neg_idx[k]])
            if train_label[j] != train_label[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1
