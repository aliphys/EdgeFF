from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT.parent / 'data'


def get_transform(dataset_name):
    if dataset_name == 'MNIST':
        return Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
        ]), False
    if dataset_name == 'FMNIST':
        return Compose([
            ToTensor(),
            Normalize((0.2860,), (0.3530,)),
            Lambda(lambda x: torch.flatten(x))
        ]), False
    if dataset_name == 'SVHN':
        return Compose([
            ToTensor(),
            Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            Lambda(lambda x: torch.flatten(x))
        ]), True
    if dataset_name == 'CIFAR10':
        return Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            Lambda(lambda x: torch.flatten(x))
        ]), True

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset(dataset_name, train=False, batch_size=512, data_root=None):
    data_root = Path(data_root) if data_root is not None else DATA_ROOT
    transform, is_color = get_transform(dataset_name)

    if dataset_name == 'MNIST':
        dataset = MNIST(str(data_root), train=train, download=True, transform=transform)
    elif dataset_name == 'FMNIST':
        dataset = FashionMNIST(str(data_root), train=train, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        dataset = SVHN(str(data_root), split='test' if not train else 'train', download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR10(str(data_root), train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, is_color


def get_test_loader(dataset_name='MNIST', batch_size=512, data_root=None):
    dataset, is_color = get_dataset(dataset_name, train=False, batch_size=batch_size, data_root=data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, is_color


def get_train_loader(dataset_name='MNIST', batch_size=512, data_root=None):
    dataset, is_color = get_dataset(dataset_name, train=True, batch_size=batch_size, data_root=data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, is_color
