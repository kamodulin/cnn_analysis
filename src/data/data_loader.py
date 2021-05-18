from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_validation_set():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    validation_set = datasets.ImageNet(root="data/imagenet/", split="val", transform=preprocess)

    loader = DataLoader(validation_set, batch_size=2400, shuffle=True, num_workers=6)

    return loader