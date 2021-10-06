import datetime
import time
import torch
import torchvision
import os

from data_loader import data_loader
from inference import predict
from metrics import accuracy_score
from utils import progress_bar


def train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler, train_loader, val_loader, device):
    model.train()

    running_loss = 0
    
    y_true = torch.empty(0, device=device)
    y_top1_pred = torch.empty(0, device=device)
    y_top5_pred = torch.empty(0, device=device)

    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, top1_pred = torch.topk(probabilities, 1)
        _, top5_pred = torch.topk(probabilities, 5)
    
        y_true = torch.cat((y_true, labels), 0)
        y_top1_pred, y_top5_pred = torch.cat((y_top1_pred, top1_pred), 0), torch.cat((y_top5_pred, top5_pred), 0)
        batch_acc1, batch_acc5 = accuracy_score(labels, top1_pred), accuracy_score(labels, top5_pred)
        avg_acc1, avg_acc5 = accuracy_score(y_true, y_top1_pred), accuracy_score(y_true, y_top5_pred)

        progress_bar((i + 1) / len(train_loader), batch_acc1=batch_acc1, batch_acc5=batch_acc5, avg_acc1=avg_acc1, avg_acc5=avg_acc5, batch_loss=running_loss / y_true.size(0))

    lr_scheduler.step()

    if args.save:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }, f"{args.save}/epoch{epoch}.pth")

    val_acc1 = accuracy_score(*predict(model, val_loader, device, topk=1))
    val_acc5 = accuracy_score(*predict(model, val_loader, device, topk=5))

    epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nEpoch duration: {epoch_time_str} train_acc1: {avg_acc1:.3f} train_acc5: {avg_acc5:.3f} train_loss: {running_loss / len(train_loader.dataset):.3f} val_acc1: {val_acc1:.3f} val_acc5: {val_acc5:.3f}")


def train(model, epochs, criterion, optimizer, lr_scheduler, train_loader, val_loader, device):
    if args.save:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": 0
        }, f"{args.save}/epoch0.pth")

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler, train_loader, val_loader, device)


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
    parser.add_argument("--save", default="", help="model weights save directory e.g. ~/data/model-weights/alexnet-train")

    args = parser.parse_args()

    train_loader, val_loader, num_classes = data_loader(args.dataset.lower(), args.batch_size, args.workers, args.num_classes)
    assert num_classes >= 5, "num_classes must be greater than or equal to 5"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
    model.to(device)

    if args.transfer:
        current = model.state_dict()
        checkpoint = torch.load(args.transfer)["model"]
        filtered = {name: tensor for name, tensor in checkpoint.items() if name in current and tensor.size() == current[name].size()}
        model.load_state_dict(filtered)

    if args.save:
        if os.path.isdir(args.save):
            import shutil
            shutil.rmtree(args.save)
        os.mkdir(args.save)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, args.epochs, criterion, optimizer, lr_scheduler, train_loader, val_loader, device)