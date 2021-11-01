import sys

from collections import OrderedDict

def get_state_dict(model):
    return OrderedDict([(name, tensor.clone()) for name, tensor in model.state_dict().items()])

def set_state_dict(model, state_dict):
    model.load_state_dict(state_dict)

def get_params(model, layer):
    state_dict = get_state_dict(model)
    W = state_dict[layer + ".weight"].clone()
    b = state_dict[layer + ".bias"].clone()
    return W, b

def set_params(model, layer, W, b):
    state_dict = get_state_dict(model)
    state_dict[layer + ".weight"] = W
    state_dict[layer + ".bias"] = b
    set_state_dict(model, state_dict)

def progress_bar(percent, **kwargs):
    bar_length = 40

    x = int(bar_length * percent)
    y = bar_length - x

    bar = f"[{'=' * x}{' ' * y}] {percent*100:>3.0f}%"
    
    for key, value in kwargs.items():
        bar += f" - {key}: {value:.3f}"
    
    sys.stdout.write("\r")
    sys.stdout.write(bar)
    sys.stdout.flush()