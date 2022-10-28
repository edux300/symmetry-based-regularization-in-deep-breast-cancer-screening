#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:39:03 2022

@author: emcastro
"""
from matplotlib import pyplot as plt
import numpy as np

def plot(data=None):
    plt.close("all")
    if data is None:
        x = [np.linspace(0, 1, 100), np.linspace(0, 2, 100)]
        y = [np.sin(x[0]*np.pi/2), np.cos(x[1]*np.pi/2)]
        yl = None
        x_label = "epoch"
        y_label = "accuracy"
        title = "Epoch vs accuracy"
        save_path = "/home/emcastro/temp.png"
        color = ["#1313F3","#AAAAAA"]
        labels = ["this is optional", ""]
        
    else:
        x = data["x"]
        y = data["y"]
        yl = data["yl"]
        
        x_label = data["x_label"]
        y_label = data["y_label"]
        title = data["title"]
        save_path = data["save_path"]
        color = data["color"]
        labels = data["labels"]

    if yl is None:
        m, M = min([min(s) for s in y]), max([max(s) for s in y])
        margin = (M-m)*0.1
        yl = (m-margin, M+margin)

    legend = dict()
    keys = []
    for i, l in enumerate(labels):
        if l != "":
            legend[l] = color[i]
            keys.append(l)
        
        
    with plt.style.context(['science', 'vibrant']):
        for xarray, yarray, c in zip(x, y, color):
            plt.plot(xarray, yarray, c=c)

        plt.ylim(*yl)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        handles = [plt.Rectangle((0,0),1,1, color=legend[k]) for k in keys]
        plt.legend(handles, keys)

        plt.savefig(save_path)

if __name__=="__main__":
    """
    data = {"x": [],
            "y": [],
            "yl": (),
            "x_label":,
            "y_label":,
            "title":,
            "save_path":,
            "color":,
            "labels":}
    """
    plot()

