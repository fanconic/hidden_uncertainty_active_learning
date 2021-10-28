import torch
from torch.utils.data import TensorDataset


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


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
