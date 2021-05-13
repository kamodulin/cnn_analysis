import torch

from torchvision import models

def get_model():
    return models.alexnet(pretrained=True)