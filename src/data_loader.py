import torch
from torchvision import datasets, transforms

def load_data():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    validation_set = datasets.ImageNet(root="../data/", split="val", transform=preprocess)

    loader = torch.utils.data.DataLoader(validation_set, batch_size=2000, shuffle=True, num_workers=2)

    return loader
