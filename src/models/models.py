import torch

from torchvision import models

from data.data_loader import load_validation_set

class BaseNet:
    def __init__(self, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        
    def _get_state(self):
        return self.model.state_dict()

    def _set_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _get_layer(self, layer):
        return self.layers[layer]

    def get_params(self, layer):
        state_dict = self._get_state()
        layer = self._get_layer(layer)

        W = state_dict[layer["weight"]]
        b = state_dict[layer["bias"]]

        return W, b

    def set_params(self, layer, new_W, new_b):
        state_dict = self._get_state()
        layer = self._get_layer(layer)
        
        state_dict[layer["weight"]] = new_W
        state_dict[layer["bias"]] = new_b

        self._set_state(state_dict)

    def predict():
        model = self.model

        model.to(self.device)
        model.eval()

        validation_set = load_validation_set()

        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in validation_set:
                images, labels = images.to(device), labels.to(device)
                
                output = model(images)
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.topk(probabilities, 5)
                
                total += labels.size(0)
                
                for i in range(labels.size(0)):
                    correct += int(labels[i] in predicted[i])

            torch.cuda.empty_cache()

        return correct, total


class AlexNet(BaseNet):
    def __init__(self, pretrained=False):
        super(AlexNet, self).__init__(models.alexnet(pretrained))

        self.layers = {
            "conv1": {
                "weight": "features.0.weight",
                "bias": "features.0.bias"
            },
            "conv2": {
                "weight": "features.3.weight",
                "bias": "features.3.bias"
            },
            "conv3": {
                "weight": "features.6.weight",
                "bias": "features.6.bias"
            },
            "conv4": {
                "weight": "features.8.weight",
                "bias": "features.8.bias"
            },
            "conv5": {
                "weight": "features.10.weight",
                "bias": "features.10.bias"
            },
            "dense1": {
                "weight": "classifier.1.weight",
                "bias": "classifier.1.bias"
            },
            "dense2": {
                "weight": "classifier.4.weight",
                "bias": "classifier.4.bias"
            },
            "dense3": {
                "weight": "classifier.6.weight",
                "bias": "classifier.6.bias"
            }
        }