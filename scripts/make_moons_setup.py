    """make a script for a part of notebook, related to all the steps above: 
       connect libraries
       download data set
       make dataframe
    """
import torch
import requests
from pathlib import Path 
import torchvision
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from torch import nn

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

#import data set and selrct samples
from sklearn.datasets import make_moons

n_samples = 1000
X, y = make_moons(n_samples,
                    noise = 0.03,
                    random_state =42)

X[:5], y[:5]

#make dataframe
moons = pd.DataFrame({"X1": X[:,0], 
                      "X2": X[:, 1],
                      "label" : y})
moons.head()
