import datetime
import time
import torch
import torchvision
import os

from data_loader import load_data
from metrics import accuracy_score
from utils import progress_bar


def train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler, data_loader, device):
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch
    }, f"{args.save}/epoch{epoch}.pth")

    epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nEpoch duration: {epoch_time_str} - Epoch acc5: {accuracy_score(y_true, y_pred):.3f} - Epoch loss: {running_loss / len(data_loader.dataset):.3f}")


def train(model, epochs, criterion, optimizer, lr_scheduler, data_loader, device):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": 0
    }, f"{args.save}/epoch0.pth")

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler, data_loader, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="alexnet", help="model")
    parser.add_argument("--dataset", default="imagenet", help="dataset")
    parser.add_argument("--epochs", default=90, type=int, help="total number of epochs")
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--pretrained", action="store_true", help="use pretrained model")
    parser.add_argument("--num-classes", default=0, type=int, help="number of classses (default: 0 = all classes in dataset")
    parser.add_argument("--transfer", default="", help="model weights for transfer learning")
    parser.add_argument("--workers", default=15, type=int, help="number of data loading workers")
    parser.add_argument("--save", default="./data/model-weights/", help="model weights save directory")

    args = parser.parse_args()

    data_loader = load_data(args.dataset.lower(), split="train", batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = args.num_classes if args.num_classes else len(data_loader.classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    model.to(device)

    if args.transfer:
        checkpoint = torch.load(args.transfer)
        model.load_state_dict(checkpoint["model"])

    if os.path.isdir(args.save):
        import shutil
        shutil.rmtree(args.save)
    
    os.mkdir(args.save)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, args.epochs, criterion, optimizer, lr_scheduler, data_loader, device)