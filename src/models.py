import torch

from torchvision import models

from data_loader import load_data

class BaseNet:
    def __init__(self, model):
        self.model = model

    def get_state(self):
        return self.model.state_dict()

    def set_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_layer(self, layer):
        return self.layers[layer]

    def get_params(self, layer):
        state_dict = self.get_state()
        layer = self.get_layer(layer)

        W = state_dict[layer["weight"]]
        b = state_dict[layer["bias"]]

        return W, b

    def set_params(self, layer, new_W, new_b):
        state_dict = self.get_state()
        layer = self.get_layer(layer)
        
        state_dict[layer["weight"]] = new_W
        state_dict[layer["bias"]] = new_b

        self.set_state(state_dict)


class AlexNet(BaseNet):
    def __init__(self, pretrained=False):
        super(AlexNet, self).__init__(models.alexnet())

        if pretrained:
            self.set_pretrained_weights()

        self.name = "alexnet"
        self.layers = { 
            "conv1": {"weight": "features.0.weight", "bias": "features.0.bias"},
            "conv2": {"weight": "features.3.weight", "bias": "features.3.bias"},
            "conv3": {"weight": "features.6.weight", "bias": "features.6.bias"},
            "conv4": {"weight": "features.8.weight", "bias": "features.8.bias"},
            "conv5": {"weight": "features.10.weight", "bias": "features.10.bias"},
            "dense1": {"weight": "classifier.1.weight", "bias": "classifier.1.bias"},
            "dense2": {"weight": "classifier.4.weight", "bias": "classifier.4.bias"},
            "dense3": {"weight": "classifier.6.weight", "bias": "classifier.6.bias"}
        }
        
    def set_pretrained_weights(self):
        weights = torch.load("~/data/model-weights/alexnet-pytorch-pretrained.pth")
        self.set_state(weights)