import torch


def predict(model, data_loader, device, topk):
    model.eval()

    y_true = torch.empty(0, device=device)
    y_pred = torch.empty(0, device=device)

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            
            probabilities = torch.nn.functional.softmax(output, dim=1) # type: ignore
            _, pred = torch.topk(probabilities, topk)

            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, pred), 0)

    return y_true, y_pred