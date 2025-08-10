import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_plot(path: str, steps, losses):
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Training loss')
    plt.title('Week 1: Bigram LM loss')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
