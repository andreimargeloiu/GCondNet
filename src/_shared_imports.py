import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import math
import re
import datetime
from joblib import dump, load
from typing import DefaultDict
import argparse
from ast import literal_eval, parse

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.utils import shuffle

from scipy.spatial.distance import cosine as cosine_distance


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed