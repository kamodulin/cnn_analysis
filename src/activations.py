import collections
import numpy as np
import torch

from data_loader import load_data
from datetime import datetime
from functools import partial
from inference import predict
from models import AlexNet


def hook_fn(name, activations, module, input, output):
    activations[name].append(output.to("cpu"))


def set_hooks(model, activations):
    names = ["dense1", "dense2", "dense3"]
    layers = [model.classifier[1], model.classifier[4], model.classifier[6]]

    for name, layer in zip(names, layers):
        layer.register_forward_hook(partial(hook_fn, name, activations))


def get_activations(net, data_loader, device):
    model = net.model
    model.to(device)

    activations = collections.defaultdict(list)

    set_hooks(model, activations)

    predict(model, data_loader, device)
    torch.cuda.empty_cache()

    for _ in activations:
        activations[_] = torch.cat(activations[_])

    return activations


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 2400
    num_workers = 6
    data_loader = load_data(split="val", batch_size=batch_size, num_workers=num_workers)

    root = "data/model-weights/pytorch-vision-classification/run_1"

    for num in range(0, 90):
        print(f"model_{num}")

        net = AlexNet()
        net.set_weights(f"{root}/model_{num}.pth")

        activations = get_activations(net, data_loader, device)

        timestamp = datetime.now().strftime("%Y%m%d%I%M%S")
        torch.save(activations, f"data/activations/model_{num}_all_dense-{timestamp}.pth")