import datetime
import time
import torch

from tqdm import tqdm

from data_loader import load_data
from metrics import accuracy_score
from models import AlexNet


def train_one_epoch(model, epoch, batch_size, criterion, optimizer, lr_scheduler, data_loader, device):
    model.train()

    training_loss = 0
    
    y_true = torch.empty(device=device)
    y_pred = torch.empty(device=device)

    start_time = time.time()

    for images, labels in tqdm(data_loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.topk(probabilities, 5)

        y_true = torch.cat((y_true, labels), 0)
        y_pred = torch.cat((y_pred, pred), 0)

    lr_scheduler.step()

    torch.save({
    'model': model.get_state(),
    'epoch': epoch
    }, f"~/data/model-weights/{model.name}-epoch{epoch}.pth")

    epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Epoch duration: {epoch_time_str} - loss: {training_loss / batch_size:.3f} - acc: {accuracy_score(y_true, y_pred):.3f}")


def train(model, epochs, batch_size, criterion, optimizer, lr_scheduler, num_workers, device):
    data_loader = load_data(split="train", batch_size=batch_size, num_workers=num_workers)

    torch.save({
        'model': model.get_state(),
        'epoch': 0
    }, f"~/data/model-weights/{model.name}-base.pth")

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch/epochs}")
        train_one_epoch(model, epoch, batch_size, criterion, optimizer, lr_scheduler, data_loader, num_workers, device)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = AlexNet()
    model = net.model
    model.to(device)

    epochs = 90
    batch_size = 256
    num_workers = 15
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, epochs, batch_size, criterion, optimizer, lr_scheduler, num_workers, device)