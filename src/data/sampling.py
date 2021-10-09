import torch
from torch.utils.data import Dataset, TensorDataset


def sampleFromClass(ds, k):
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in ds:
        c = label
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_data.append(torch.unsqueeze(torch.tensor(data), 0))
            train_label.append(torch.unsqueeze(torch.tensor(label), 0))
        else:
            test_data.append(torch.unsqueeze(torch.tensor(data), 0))
            test_label.append(torch.unsqueeze(torch.tensor(label), 0))

    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label)
    test_data = torch.cat(test_data)
    test_label = torch.cat(test_label)

    return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )
