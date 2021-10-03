from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(dataset, split, batch_size, num_workers):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "imagenet":
        data = datasets.ImageNet(root="~/data/datasets/imagenet", split=split, transform=preprocess)
    elif dataset == "cifar10":
        data = datasets.CIFAR10(root="~/data/datasets/CIFAR10", train=True if split == "train" else False, transform=preprocess)
    else:
        raise AssertionError(f"Invalid dataset {dataset}")
    
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)    

    return loader
