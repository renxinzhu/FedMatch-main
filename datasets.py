import torch
from torch.utils.data import Subset, DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize
from randaugment import RandAugment
import random

LABEL_PER_CLIENT = 5


def _read_FashionMNIST(noised=False):
    transform = Compose([RandAugment(3, 4), ToTensor()]
                        ) if noised else ToTensor()
    return FashionMNIST(root="data", train=True,
                        download=True, transform=transform)


def _read_test_FashionMNIST():
    dataset = FashionMNIST(root="data", train=False,
                           download=True, transform=ToTensor())
    return Subset(dataset, range(500))


normalization_values_CIFAR10 = (
    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))


def _read_CIFAR10(noised=False):
    transform = Compose([
        RandAugment(3, 4),
        ToTensor(),
        Normalize(*normalization_values_CIFAR10),
    ]) if noised else ToTensor()
    return CIFAR10(root="data", train=True,
                        download=True, transform=transform)


def _read_test_CIFAR10():
    transform = Compose([ToTensor(), Normalize(*normalization_values_CIFAR10)])
    dataset = CIFAR10(root="data", train=True,
                      download=True, transform=transform)
    return Subset(dataset, range(500))


datasets = {
    'cifar10': _read_CIFAR10,
    'fashionMNIST': _read_FashionMNIST,
}

test_dataset = {
    'cifar10': _read_test_CIFAR10,
    'fashionMNIST': _read_test_FashionMNIST,
}


class DatasetWithAndWithoutNoised(Dataset):
    def __init__(self, dataset_key: str):
        read_dataset = datasets[dataset_key]
        self.dataset = read_dataset(False)
        self.noisedDataset = read_dataset(True)
        self.length = len(self.dataset)

    @property
    def train_labels(self):
        return self.dataset.targets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        noised_X, _ = self.noisedDataset[idx]
        return noised_X, X, y


def _split_noniid_datasets(dataset: Dataset, num_clients: int, label_per_client: int):
    # sort dataset by labels
    train_labels = torch.Tensor([y for *_, y in dataset])
    sorted_idxes = torch.argsort(train_labels).tolist()
    sorted_dataset = Subset(dataset, sorted_idxes)

    # partition the sorted dataset by order
    num_partitions = num_clients*label_per_client
    size = len(dataset) // (num_partitions)
    ranges = [range(size*i, size*(i+1))
              for i in range(num_partitions)]
    random.shuffle(ranges)

    partition_ranges = []
    for idx in range(0, num_partitions, label_per_client):
        temp_range = []
        for addition in range(label_per_client):
            temp_range += ranges[idx + addition]
        partition_ranges.append(temp_range)

    return [Subset(sorted_dataset, _range)
            for _range in partition_ranges]


def generate_random_dataloader(dataset_key: str):
    dataset = datasets[dataset_key]()
    size = len(dataset) // 50
    idxes = torch.randint(0, len(dataset), (size,)).tolist()
    subset = Subset(dataset, idxes)

    return DataLoader(subset, batch_size=64, shuffle=True)


def generate_random_dataloaders(dataset_key: str, label_ratio: float = 0.05):
    dataset = datasets[dataset_key]()
    size = len(dataset) // 50
    labeled_size = int(size * label_ratio)
    idxes = torch.randint(0, len(dataset), (size,)).tolist()
    subset = Subset(dataset, idxes)

    return {
        'labeled': DataLoader(Subset(subset, range(labeled_size)), batch_size=64, shuffle=True),
        'unlabeled': DataLoader(Subset(subset, range(labeled_size, size)), batch_size=64, shuffle=True)
    }


def generate_dataloaders(dataset_key: str, num_clients: int, label_ratio: float, iid: bool, batch_size_s: int, batch_size_u: int):
    dataset = DatasetWithAndWithoutNoised(dataset_key)
    size = len(dataset) // num_clients
    labeled_size = int(size * label_ratio)
    final_dataset = Subset(dataset, range(size * num_clients))
    datasets = random_split(final_dataset, [size for _ in range(
        num_clients)]) if iid else _split_noniid_datasets(final_dataset, num_clients, LABEL_PER_CLIENT)

    return [{
        'labeled': DataLoader(Subset(ds, range(labeled_size)), batch_size=batch_size_s, shuffle=True),
        'unlabeled': DataLoader(Subset(ds, range(labeled_size, size)), batch_size=batch_size_u, shuffle=True)
    } for ds in datasets]


def generate_little_dataloaders(dataset_key: str, num_clients: int, label_ratio: float, iid: bool, size: int):
    dataset = DatasetWithAndWithoutNoised(dataset_key)
    dataset_size = len(dataset) // num_clients
    labeled_size = int(size * label_ratio)
    final_dataset = Subset(dataset, range(num_clients * dataset_size))
    datasets = random_split(final_dataset, [dataset_size for _ in range(
        num_clients)]) if iid else _split_noniid_datasets(final_dataset, num_clients, LABEL_PER_CLIENT)

    return [{
        'labeled': DataLoader(Subset(ds, range(labeled_size)), batch_size=8, shuffle=True),
        'unlabeled': DataLoader(Subset(ds, range(labeled_size, size)), batch_size=32, shuffle=True)
    } for ds in datasets]


def generate_test_dataloader(dataset_key: str):
    dataset = test_dataset[dataset_key]()
    return DataLoader(dataset, batch_size=64, shuffle=True)


def generate_little_test_dataloader(dataset_key: str):
    dataset = test_dataset[dataset_key]()
    return DataLoader(Subset(dataset, range(100)), batch_size=16, shuffle=True)
