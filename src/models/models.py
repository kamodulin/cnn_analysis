import torch

from torchvision import models

from data.data_loader import load_data

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

    def train(self):
        import datetime
        import time
        from tqdm import tqdm

        self.model.to(self.device)
        
        epochs = 90
        batch_size = 32
        training_set = load_data("train", batch_size=batch_size, num_workers=8)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        checkpoint = {
            'model': self._get_state(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': 0
            }

        torch.save(checkpoint, f"data/model-weights/{self.name}-base.pth")
        
        for epoch in range(1, epochs+1):

            print(f"Epoch {epoch}/{epochs}")
            start_time = time.time()
            
            self.model.train()
            
            training_loss = 0
            correct, total = 0, 0

            for images, labels in tqdm(training_set, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.topk(probabilities, 5)
                
                total += labels.size(0)
                
                for i in range(labels.size(0)):
                    correct += int(labels[i] in predicted[i])
            
            lr_scheduler.step()

            checkpoint = {
                'model': self._get_state(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
                }

            torch.save(checkpoint, f"data/model-weights/{self.name}-epoch{epoch}.pth")

            epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            print(f"Epoch duration: {epoch_time_str} - loss: {training_loss / 100:.3f} - acc: {correct / total:.3f}")

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        validation_set = load_data("val", batch_size=2400, num_workers=6)

        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in validation_set:
                images, labels = images.to(self.device), labels.to(self.device)
                
                output = self.model(images)
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.topk(probabilities, 5)
                
                total += labels.size(0)
                
                for i in range(labels.size(0)):
                    correct += int(labels[i] in predicted[i])

            torch.cuda.empty_cache()

        return correct, total


class AlexNet(BaseNet):
    def __init__(self, pretrained=False):
        super(AlexNet, self).__init__(models.alexnet())

        if pretrained:
            weights = torch.load("data/model-weights/alexnet-pytorch-pretrained.pth") #only model_state_dict
            self._set_state(weights)

        self.name = "alexnet"

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