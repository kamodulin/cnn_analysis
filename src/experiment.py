import os
import torch

from collections import OrderedDict
from datetime import datetime
from itertools import product
from tqdm import tqdm

from data_loader import load_data
from inference import predict
from metrics import accuracy_score
from models import AlexNet, VGG16
from perturb import knockout


class Experiment:
    def __init__(self, layer, level, fraction, repeat):
        self.layer = layer
        self.level = level
        self.fraction = fraction
        self.repeat = repeat

    def run(self, net, data_loader, device):
        net = net.reset_weights()
        knockout(net, self.layer, self.level, self.fraction)

        net.model.to(device)
        y_true, y_pred = predict(net.model, data_loader, device)
        accuracy = accuracy_score(y_true, y_pred)

        torch.cuda.empty_cache()

        return accuracy


class Manager:
    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime("%Y%m%d%I%M%S")
        self.csv = self.init_files()
        self.expts = self.make_expts(params)

    def make_expts(self, params):
        expts = []

        for value in product(*params.values()):
            e = Experiment(*value)
            e.manager_file = self.csv
            expts.append(e)

        return expts

    def init_files(self):
        basedir = "data/perturbation-experiments/"
        
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        csv = f"{basedir}/{self.timestamp}-experiments.csv"
        meta = f"{basedir}/{self.timestamp}-metadata.txt"

        with open(meta, "w") as f:
            for key, value in self.params.items():
                f.write(f"{str(key)}: {str(value)}\n")
        
        with open(csv, "w") as f:
            f.write("layer, level, fraction, repeat, accuracy\n")
        
        return csv

    def run(self, net, data_loader, device):
        for e in tqdm(self.expts, desc="experiment"):
            
            accuracy = e.run(net, data_loader, device)

            with open(self.csv, "a") as f:
                f.write(f"{e.layer}, {e.level}, {e.fraction}, {e.repeat}, {accuracy}\n")
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # alexnet
    # params = OrderedDict(
    #     layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "dense1", "dense2", "dense3"],
    #     levels = ["synapse", "node"],
    #     fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     repeats = range(3)
    # )

    # vgg16
    params = OrderedDict(
        layers = ["conv1-1",
                  "conv1-2",
                  "conv2-1",
                  "conv2-2",
                  "conv3-1",
                  "conv3-2",
                  "conv3-3",
                  "conv4-1",
                  "conv4-2",
                  "conv4-3",
                  "conv5-1",
                  "conv5-2",
                  "conv5-3",
                  "dense1",
                  "dense2",
                  "dense3"],
        levels = ["synapse", "node"],
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        repeats = range(1)
    )

    batch_size = 256
    num_workers = 15
    data_loader = load_data(split="val", batch_size=batch_size, num_workers=num_workers)
    
    net = VGG16(pretrained=True)

    m = Manager(params)
    m.run(net, data_loader, device)