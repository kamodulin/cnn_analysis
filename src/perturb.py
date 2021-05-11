import random
import torch

from state import get_params, set_params

def _random_indices(n, percent):
    idx = random.sample(range(n), round(n * percent))
    return idx

def synapse_knockout(W, b, percent):
    new_params = []

    for params in [W, b]:
        n_params = params.numel()
        mask = torch.ones(n_params)
        idx = _random_indices(n_params, percent)
        mask[idx] = 0

        new_params.append(params * mask.reshape(params.shape))

    return new_params

def node_knockout(W, b, percent):
    new_W = W.clone()
    new_b = b.clone()
    
    n_nodes = len(b)
    idx = _random_indices(n_nodes, percent)

    if len(W.shape) == 2:
        # dense layers
        new_W[:, idx] = 0
    else:
        # conv layers
        new_W[idx, :] = 0

    new_b[idx] = 0

    return new_W, new_b

def knockout(model, layer, percent, level="synapse"):
    if level == "synapse":
        f = synapse_knockout
    else:
        f = node_knockout

    W, b = get_params(model, layer)
    new_W, new_b = f(W, b, percent)
    set_params(model, layer, new_W, new_b)