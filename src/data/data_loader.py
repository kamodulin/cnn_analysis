from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(split, batch_size, num_workers=6):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data = datasets.ImageNet(root="data/imagenet/", split=split, transform=preprocess)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return loader