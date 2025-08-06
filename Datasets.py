from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
class GetDataset(Dataset):
    def __init__(self, data, labels, real_labels):
        self.data = data
        self.labels = labels
        self.real_labels = real_labels

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        real_label = np.int64(self.real_labels[index])
        return fea0, fea1, label, real_label

    def __len__(self):
        return len(self.labels)


class GetAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        return fea0, fea1, label, class_labels0, class_labels1

    def __len__(self):
        return len(self.labels)
