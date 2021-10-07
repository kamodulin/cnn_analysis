import random
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SubsetResetIndex(Subset):
    def __init__(self, dataset, indices, mapping):
        self.dataset = dataset
        self.indices = indices
        self.mapping = mapping

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        label = self.mapping[item[1]]
        return (item[0], label)

    def __len__(self):
        return len(self.indices)


def load_dataset(dataset, split):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "imagenet":
        data = datasets.ImageNet(root="~/data/datasets/imagenet", split=split, transform=preprocess)
    elif dataset == "cifar10":
        data = datasets.CIFAR10(root="~/data/datasets/CIFAR10", train=True if split == "train" else False, transform=preprocess, download=True)
    elif dataset == "cifar100":
        data = datasets.CIFAR100(root="~/data/datasets/CIFAR100", train=True if split == "train" else False, transform=preprocess, download=True)
    else:
        raise AssertionError(f"Invalid dataset {dataset}")

    return data


# add seed for deterministic choice?
def data_loader(dataset, batch_size, num_workers, num_classes=0):
    train_data = load_dataset(dataset, "train")
    val_data = load_dataset(dataset, "val")

    total_num_classes = len(train_data.classes)

    if not num_classes:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        num_classes = total_num_classes
        
    else:
        assert num_classes <= total_num_classes, "num_classes cannot exceed "

        targets = random.sample(range(total_num_classes), num_classes)
        mapping = {x:i for i, x in enumerate(targets)} 

        train_data_subset = create_subset(train_data, targets, mapping)
        val_data_subset = create_subset(val_data, targets, mapping)
        
        train_loader = DataLoader(train_data_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    return train_loader, val_loader, num_classes


def create_subset(data, targets, mapping):
    idx = [i for i, label in enumerate(data.targets) if label in targets]
    subset = SubsetResetIndex(data, idx, mapping)
    return subset