import datetime
import time
import torch
import os

from metrics import accuracy_score
# from models import AlexNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import progress_bar

def load_data(split, batch_size, num_workers):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data = datasets.ImageNet(root="~/data/datasets/imagenet", split=split, transform=preprocess)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader

def train_one_epoch(model, epoch, batch_size, criterion, optimizer, lr_scheduler, data_loader, device):
    model.train()

    running_loss = 0
    
    y_true = torch.empty(0, device=device)
    y_pred = torch.empty(0, device=device)

    start_time = time.time()

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.topk(probabilities, 5)
    
        y_true = torch.cat((y_true, labels), 0)
        y_pred = torch.cat((y_pred, pred), 0)

        progress_bar((i + 1) / len(data_loader), avg_acc5=accuracy_score(y_true, y_pred), batch_acc5=accuracy_score(labels, pred), batch_loss=running_loss / y_true.size(0))

    lr_scheduler.step()

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }, f"{SAVE_PATH}/epoch{epoch}.pth")
    
    epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nEpoch duration: {epoch_time_str} - Epoch acc5: {accuracy_score(y_true, y_pred):.3f} - Epoch loss: {running_loss / len(data_loader.dataset):.3f}")


def train(model, epochs, batch_size, criterion, optimizer, lr_scheduler, num_workers, device):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': 0
    }, f"{SAVE_PATH}/epoch0.pth")

    data_loader = load_data(split="train", batch_size=batch_size, num_workers=num_workers)

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_one_epoch(model, epoch, batch_size, criterion, optimizer, lr_scheduler, data_loader, device)


if __name__ == "__main__":
    SAVE_PATH = "/home/kaa716/data/model-weights/alexnet-imagenet-train"
    
    if os.path.isdir(SAVE_PATH):
        import shutil
        shutil.rmtree(SAVE_PATH)
    
    os.mkdir(SAVE_PATH)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net = AlexNet()
    model = models.alexnet()
    # model.classifier[6] = torch.nn.Linear(4096, 10) #CIFAR10
    # net.set_weights("/home/kaa716/data/model-weights/alexnet-cifar10-training-scratch/epoch89.pth")
    # model.classifier[6] = torch.nn.Linear(4096, 1000) #change to ImageNet num_classes
    
    model.to(device)

    epochs = 90
    batch_size = 256
    num_workers = 15
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, epochs, batch_size, criterion, optimizer, lr_scheduler, num_workers, device)
