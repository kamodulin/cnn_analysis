import csv
import time
import torch

from tqdm import tqdm

# from inference import predict
from layers import layers
from model import get_model
from perturb import knockout

percentages = [x / 10 for x in range(0, 11)]
repeats = 1

if __name__ == "__main__":
    with open("experiments.csv", "a") as f:
        writer = csv.writer(f, delimiter=',')
        for run in tqdm(range(repeats), desc="run"):
            for percent in tqdm(percentages, desc="percent", leave=False):
                for layer in tqdm(layers.keys(), desc="layers", leave=False):            
                    model = get_model() # get a fresh AlexNet each iteration
                    knockout(model, layers[layer], percent)
                    accuracy = predict(model)
                    
                    experiment = f'{layer}_run{str(run)}_p{str(percent)}'
                    writer.writerow([experiment, accuracy])