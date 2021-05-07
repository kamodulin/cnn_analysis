import tqdm
import torch

from data_loader import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loader = load_data()

def predict(model):
    total, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.topk(probabilities, 5)
            
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                correct += int(labels[i] in predicted[i])

    torch.cuda.empty_cache()

    return correct/total