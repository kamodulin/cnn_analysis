def get_state(model):
    return model.state_dict()

def set_state(model, state_dict):
    model.load_state_dict(state)

def get_params(model, layer):
    state_dict = get_state(model)
    
    W = state_dict[layer["weight"]]
    b = state_dict[layer["bias"]]

    return W, b

def set_params(model, layer, new_W, new_b):
    state_dict = get_state(model)

    W = state_dict[layer["weight"]]
    b = state_dict[layer["bias"]]

    state_dict[W] = new_W
    state_dict[b] = new_b

    set_state(model, state_dict)