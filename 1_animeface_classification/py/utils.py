# -*- coding: utf-8 -*-
import random
import os

import numpy as np

# code from https://www.kaggle.com/bminixhofer/
#           deterministic-neural-networks-using-pytorch
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

