import argparse
import os
import pandas as pd

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
        df = pd.DataFrame(data=[self.layer, self.level, self.fraction, self.repeat])
        df.to_csv(self.filepath, mode='a', header=False)

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
        basedir = os.path.dirname("data/perturbation-experiments/")
        
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        csv = f"{basedir}/{self.timestamp}-experiments.csv"
        meta = f"{basedir}/{self.timestamp}-metadata.txt"

        with open(meta, "w") as f:
            for key, value in self.params.items():
                f.write(f"{str(key)}: {str(value)}\n")
        
        with open(csv, "w") as f:
            f.write("layer, level, fraction, repeat")
        
        return csv

if __name__ == "__main__":
    params = OrderedDict(
        layers = ["conv1", "conv2"],
        levels = ["synapse", "node"],
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        repeats = range(3)    
    )

    m = Manager(params)

    for expt in m.expts:
        model = AlexNet(pretrained=True)
        knockout(model, expt.layer, expt.level, expt.fraction)
        expt.accuracy(model.predict())
        expt.save()














# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-r", "--repeats", action="store", default=1)
#     parser.add_argument("-l", "--level", action="store", default="synapse")

#     args = parser.parse_args()

#     # timestamp = datetime.now().strftime("%Y%m%d_%I%M%S")
#     # logging.basicConfig(filename=f"logs/{timestamp}.log", filemode="a", format="%(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.DEBUG)
#     # logging.info(f"INFO:repeats={args.repeats}, level={args.level}")

#     # for layer in tqdm(layers.keys(), desc="layer", leave=False):
#     #     for fraction in tqdm([x / 10 for x in range(0, 11)], desc="fraction", leave=False):
#     #         for repeat in tqdm(range(int(args.repeat)), desc="repeat", leave=False):

#                 model = AlexNet(pretrained=True)
#                 knockout(model, layer, fraction, level=args.level)
#                 EXPT.accuracy_score(model.predict())

#                 accuracy = predict(model)
                
#                 experiment = f"EXPT:layer={layer}, repeat={str(repeat+1)}, fraction={str(fraction)}"

#                 logging.info(f"{experiment}, accuracy={accuracy}")