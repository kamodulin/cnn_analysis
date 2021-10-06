import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_dataset(dataset, split, batch_size, num_workers):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "imagenet":
        data = datasets.ImageNet(root="~/data/datasets/imagenet", split=split, transform=preprocess, download=True)
    elif dataset == "cifar10":
        data = datasets.CIFAR10(root="~/data/datasets/CIFAR10", train=True if split == "train" else False, transform=preprocess, download=True)
    else:
        raise AssertionError(f"Invalid dataset {dataset}")

    return data


def data_loader(dataset, batch_size, num_workers, num_classes=0):
    train_data = load_dataset(dataset, "train", batch_size, num_workers)
    val_data = load_dataset(dataset, "val", batch_size, num_workers)

    if not num_classes:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        num_classes =  len(train_loader.classes)
        
    else:
        targets = random.sample(range(len(train_data.classes)), num_classes)
        mapping = {x:i for i, x in enumerate(targets)} 

        train_data_subset = create_subset(train_data, targets, mapping)
        val_data_subset = create_subset(val_data, targets, mapping)
        
        train_loader = DataLoader(train_data_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader, num_classes


def create_subset(data, targets, mapping):
    idx = [i for i, label in enumerate(data.targets) if label in targets]
    data.targets = [mapping[x] if x in mapping.keys() else None for x in data.targets]
    subset = Subset(data, idx)
    return subset