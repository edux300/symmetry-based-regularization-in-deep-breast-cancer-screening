#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:39:03 2022

@author: emcastro
"""
from matplotlib import pyplot as plt

def plot(data=None):
    plt.close("all")
    if data is None:
        x = ["a", "b", "c"]
        y = [0.8, 0.82, 0.85]
        yl = (0.75, 0.9)
        x_label = "model"
        y_label = "accuracy"
        title = "Model vs accuracy"
        save_path = "/home/emcastro/temp.png"
        color = ["#1313F3","#AAAAAA","#202020"]
        labels = ["this is optional", "", ""]
        
    else:
        x = data["x"]
        y = data["y"]
        yl = data["yl"]
        if yl is None:
            m, M = min(y), max(y)
            margin = (M-m)*0.1
            yl = (m-margin, M+margin)
        x_label = data["x_label"]
        y_label = data["y_label"]
        title = data["title"]
        save_path = data["save_path"]
        color = data["color"]
        labels = data["labels"]

    legend = dict()
    keys = []
    for i, l in enumerate(labels):
        if l != "":
            legend[l] = color[i]
            keys.append(l)
        
        
    with plt.style.context(['science', 'vibrant']):
        plt.bar(x, y, color=color)
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
    
"""
data = {"x": ["0", "2", "3", "4", "5", "all"],
        "y": string,
        "yl": None,
        "x_label": "Group Equiv Stride", 
        "y_label": metric,
        "title": "",
        "save_path": f"/home/emcastro/{metric}.png",
        "color": color,
        "labels":["baseline","","hybrid","", "", "p4"]}
"""
