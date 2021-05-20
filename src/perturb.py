import random
import torch

def random_indices(n, fraction):
    idx = random.sample(range(n), round(n * fraction))
    return idx


def synapse_knockout(W, b, fraction):
    new_params = []

    for params in (W, b):
        n_params = params.numel()
        mask = torch.ones(n_params)
        idx = random_indices(n_params, fraction)
        mask[idx] = 0

        new_params.append(params * mask.reshape(params.shape))

    return new_params


def node_knockout(W, b, fraction):
    new_W = W.clone()
    new_b = b.clone()
    
    n_nodes = len(b)
    idx = random_indices(n_nodes, fraction)
    
    new_W[idx, :] = 0
    new_b[idx] = 0

    return new_W, new_b


def knockout(model, layer, level, fraction):
    if level == "synapse":
        f = synapse_knockout

    elif level == "node":
        f = node_knockout

    else:
        return None

    W, b = model.get_params(layer)
    new_W, new_b = f(W, b, fraction)
    model.set_params(layer, new_W, new_b)