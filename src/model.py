import torch

from torchvision import models

def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.alexnet(pretrained=True)
    model.to(device)

    return model