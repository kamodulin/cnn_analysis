import argparse
import os

from collections import OrderedDict
from datetime import datetime
from itertools import product
from tqdm import tqdm

from perturb import knockout
from models import AlexNet

class Experiment:
    def __init__(self, layer, level, fraction, repeat):
        self.layer = layer
        self.level = level
        self.fraction = fraction
        self.repeat = repeat
    
    def file_to_write(self, filepath):
        self.filepath = filepath

    def perturb(self):
        pass

    def accuracy(self, correct, total):
        self.accuracy_score = correct / total
    
    def save(self):
        with open(self.filepath, "a") as f:
            f.write(f"{self.layer}, {self.level}, {self.fraction}, {self.repeat}, {self.accuracy_score}\n")

class Manager:
    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime("%Y%m%d%I%M%S")
        self.filepath = self.create_files()
        
        self.expts = self.make_expts(params)

    def make_expts(self, params):
        expts = []

        for value in product(*params.values()):
            e = Experiment(*value)
            e.file_to_write(self.filepath)
            expts.append(e)

        return expts

    def create_files(self):
        basedir ="data/perturbation-experiments/"
        
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

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--layers", action="store", default=None)
    # parser.add_argument("--levels", action="store", default="synapse")
    # parser.add_argument("--repeats", action="store", default=1)
    # parser.add_argument("--fractions", action="store", default=None)

    # args = parser.parse_args()

    params = OrderedDict(
        # model = "alexnet",
        layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "dense1", "dense2", "dense3"],
        levels = ["synapse", "node"],
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        repeats = range(3)
    )

    m = Manager(params)

    for expt in tqdm(m.expts, desc="experiment"):
        model = AlexNet(pretrained=True)
        knockout(model, expt.layer, expt.level, expt.fraction)
        expt.accuracy(*model.predict())
        expt.save()