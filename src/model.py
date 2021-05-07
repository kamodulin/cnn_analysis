from PIL import Image
import torch
from torchvision import datasets, models, transforms

def get_net():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.alexnet(pretrained=True)
    model.eval()
    model.to(device)

    return model