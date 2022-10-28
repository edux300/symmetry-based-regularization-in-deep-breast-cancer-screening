#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:59:20 2021

@author: emcastro
"""

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os

METHOD_NAME = "contrastive01"
TRANSFORM_NAME = "rotation"
WORKING_PATH = os.environ["DEEPMM_WORKDIR"]
LOAD_PATH = WORKING_PATH
SAVE_PATH = WORKING_PAT + "/temp_results"
zs = np.load(LOAD_PATH+"zs.npy")
ids = np.load(LOAD_PATH+"ids.npy")
ys = np.load(LOAD_PATH+"ys.npy")

#%%
repre = TSNE(2, verbose=1)
zs = repre.fit_transform(zs)


#%%

def draw(x, y, t="", sort=False):
    if sort:
        idx = np.argsort(y)
        x = x[:, idx]
        y = y[idx]
    with plt.style.context(['science']):
        plt.figure()
        plt.scatter(*x, c=y, alpha=0.5)
        plt.show()
        if t!= "":
            plt.title(t)

path = f"{SAVE_PATH}/method_{METHOD_NAME}_transform_{TRANSFORM_NAME}/"
#os.makedirs(path)
for i in range(16):
    draw(zs.T, ids==i, sort=True)
    plt.savefig(path + f"example_{i}.png")
draw(zs.T, ys, "classes")
plt.savefig(path + "classes.png")
plt.close("all")

