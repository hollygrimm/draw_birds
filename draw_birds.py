"""DRAW Birds.
"""
"""
Deep Recurrent Attentive Writer (DRAW) Training code and utilities are licensed under APL2.0 from

Parag Mital
---------------------
https://github.com/pkmital/pycadl/blob/master/cadl/draw.py

Copyright 2017 Holly Grimm.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from cadl.draw import train_dataset
from cadl.dataset_utils import Dataset


def train():
    """Train DRAW on image files
    """    
    path = './birdimages_64sq/'
    fs = [
        os.path.join(path, f) for f in os.listdir(path)
        if f.endswith('.jpg')
    ]
    images = [plt.imread(f) for f in fs]
    images = np.asarray(images)
    images = images.reshape((images.shape[0], -1))
    ds = Dataset(np.r_[images[:6000]],
                split=[.8, .1, .1])

    train_dataset(ds, 64, 64, 3, n_epochs=1000)

if __name__ == '__main__':
    train()