import torch


def predict(model, data_loader, device):
    model.eval()

    y_true = torch.empty(device=device)
    y_pred = torch.empty(device=device)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.topk(probabilities, 5)

            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, pred), 0)

    return y_true, y_pred