import collections
import numpy as np
import torch

from data_loader import load_data
from functools import partial
from inference import predict
from models import AlexNet
from scipy import stats


def hook_fn(name, activations, module, input, output):
    activations[name].append(output.half().to("cpu"))


def set_hooks(model, activations):
    # names = ["conv1", "conv2", "conv3", "conv4", "conv5", "dense1", "dense2", "dense3"]
    # layers = [model.features[0], model.features[3], model.features[6], model.features[8], model.features[10], model.classifier[1], model.classifier[4], model.classifier[6]]
    
    names = ["conv2"]
    layers = [model.features[3]]

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

    batch_size = 256
    num_workers = 15
    data_loader = load_data(split="val", batch_size=batch_size, num_workers=num_workers)

    root = "/home/kaa716/data/model-weights/pytorch-vision-classification/run_1"

    final_net = AlexNet()
    final_net.set_weights(f"{root}/model_89.pth")

    A_f = get_activations(final_net, data_loader, device)
    A_f = A_f['conv2'].transpose(0, 1).reshape(A_f['conv2'].shape[1], -1).numpy()

    pearson_scores = collections.defaultdict(list)

    for num in range(0,90):

        net = AlexNet()
        net.set_weights(f"{root}/model_{num}.pth")

        A_i = get_activations(net, data_loader, device) 
        A_i = A_i['conv2'].transpose(0, 1).reshape(A_i['conv2'].shape[1], -1).numpy()

        for idx in range(len(A_i)):
            print(idx)
            
            pearson = stats.pearsonr(A_f[idx], A_i[idx])[0]
            pearson_scores['conv2'].append(pearson)
                
        np.save('conv2-pearson_scores.npy', dict(pearson_scores))
