import random
import torch

from utils import get_params, set_params


def random_indices(n, fraction):
    idx = random.sample(range(n), round(n * fraction))
    return idx


def synapse_knockout(W, b, fraction):
    new_params = []

    for params in (W, b):
        params = params.cpu()
        n_params = params.numel()
        mask = torch.ones(n_params)
        idx = random_indices(n_params, fraction)
        mask[idx] = 0

        new_params.append(torch.mul(params, mask.reshape(params.shape)))

    return new_params


def node_knockout(W, b, fraction):
    n_nodes = len(b)
    idx = random_indices(n_nodes, fraction)
    W[idx, :] = 0
    b[idx] = 0
    return W, b


def knockout(model, layer, level, fraction):
    if level == "synapse":
        f = synapse_knockout
    elif level == "node":
        f = node_knockout
    
    W, b = get_params(model, layer)
    new_W, new_b = f(W, b, fraction)
    set_params(model, layer, new_W, new_b)
