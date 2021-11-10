import os
import torch
import torchvision

from collections import OrderedDict
from datetime import datetime
from itertools import product
from torch.utils.data import DataLoader

from data_loader import load_dataset
from inference import predict
from metrics import accuracy_score
from perturb import knockout
from utils import get_state_dict, set_state_dict, progress_bar


class Experiment:
    def __init__(self, layer, level, fraction, repeat):
        self.layer = layer
        self.level = level
        self.fraction = fraction
        self.repeat = repeat

    def run(self, model, data_loader, device):
        # reset state_dict
        set_state_dict(model, pretrained_weights)
        knockout(model, self.layer, self.level, self.fraction)
        model.to(device)

        y_true, y_pred = predict(model, data_loader, device, topk=5)
        accuracy = accuracy_score(y_true, y_pred)
        torch.cuda.empty_cache()

        return accuracy


class Manager:
    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime("%Y%m%d_%I%M%S")
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
            f.write(f"{str(args)}\n\n")
            for key, value in self.params.items():
                f.write(f"{str(key)}: {str(value)}\n")

        with open(csv, "w") as f:
            f.write("layer, level, fraction, repeat, accuracy\n")

        return csv

    def run(self, model, data_loader, device):
        for i, expt in enumerate(self.expts):

            accuracy = expt.run(model, data_loader, device)

            with open(self.csv, "a") as f:
                f.write(
                    f"{expt.layer}, {expt.level}, {expt.fraction}, {expt.repeat}, {accuracy}\n"
                )

            progress_bar((i + 1) / len(self.expts))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="alexnet", help="model")
    parser.add_argument("--dataset", default="imagenet", help="dataset")
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument(
        "--load-weights",
        default="",
        help=
        "model weights to load, default is to load PyTorch pretrained weights")
    parser.add_argument("--layers",
                        nargs="+",
                        default="all",
                        help="layers to perform knockout")
    parser.add_argument("--analysis",
                        nargs="+",
                        default=["synapse", "node"],
                        help="level of analysis")
    parser.add_argument("--fraction",
                        nargs="+",
                        default=[x / 10 for x in range(0, 11)],
                        type=float,
                        help="knockout fraction")
    parser.add_argument("--repeats",
                        default=1,
                        type=int,
                        help="number of repeats")
    parser.add_argument("--workers",
                        default=15,
                        type=int,
                        help="number of data loading workers")

    args = parser.parse_args()

    val_data = load_dataset(args.dataset.lower(), split="val")
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers)
    num_classes = len(val_data.classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.load_weights:
        model = torchvision.models.__dict__[args.model](
            pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(args.load_weights)["model"]
        set_state_dict(model, checkpoint)
    else:
        model = torchvision.models.__dict__[args.model](
            pretrained=True, num_classes=num_classes)

    pretrained_weights = get_state_dict(model)

    if args.layers == "all":
        layers = []
        for name, param in model.named_parameters():
            truncated = ".".join(name.split(".")[:-1])
            if truncated not in layers:
                layers.append(truncated)
    else:
        layers = []
        for layer in args.layers:
            weight = layer + ".weight"
            bias = layer + ".bias"
            named_params = [name for name, _ in model.named_parameters()]

            if weight not in named_params or bias not in named_params:
                raise ValueError(f"{layer} is not a valid layer")
            else:
                layers.append(layer)

    params = OrderedDict(layers=layers,
                         levels=args.analysis,
                         fraction=args.fraction,
                         repeats=range(args.repeats))

    m = Manager(params)
    m.run(model, val_loader, device)
