import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from models import AlexNet
from svcca import cca_core

root = "~data/activations/"
activation_files = sorted(os.listdir(root), key=lambda x: x.split('-')[1])
A_f = torch.load(root+activation_files[-1])

scores = []

for num, activations in enumerate(activation_files):
    print(f"Comparing final activations with model_{num}")
    
    A_i = torch.load(root+activations)
    
    for layer in ['dense1', 'dense2', 'dense3']:
        print(f"------{layer}")
        
        cca = cca_core.get_cca_similarity(A_i[layer].T.cpu().detach().numpy(),
                                          A_f[layer].T.cpu().detach().numpy(),
                                          epsilon=1e-6)
        
        score = np.mean(cca["cca_coef1"])
        
        scores.append(score)
        
    np.savetxt('~/scores.csv', scores, delimiter=',')