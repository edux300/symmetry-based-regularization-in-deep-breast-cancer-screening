#%%
from generic_bar_plot import plot

string = "0,897313607367280,9020446461863470,9070450080874120,9214316047482640,9100554416028540,914119302480293"
metric = "ROC AUC"

string = string.replace("0,", ", 0.").split(",")
string = [float(x) for x in string[1:]]
color = ["#1b9e77", "#d95f02", "#7570b3", "#d95f02", "#d95f02", "#e7298a"]

data = {"x": ["0", "2", "3", "4", "5", "all"],
        "y": string,
        "yl": None,
        "x_label": "Group Equiv Stride", 
        "y_label": metric,
        "title": "",
        "save_path": f"/home/emcastro/{metric}.png",
        "color": color,
        "labels":["baseline","","hybrid","", "", "p4"]}

plot(data)

#%%
from generic_line_plot import plot
import numpy as np
from glob import glob
from os.path import join

basepath = "/media/emcastro/External_drive/results_best/Symmetry-based regularization in deep breast cancer screening/final results/3. architecture/table/"
child_paths = ["1. z2_64_scratch (z2)", "2. p4_64_scratch (p4)", "3. p4-2_64_scratch (hybrid)"]

paths = [join(basepath, p) for p in child_paths]
metric = "bal-accuracy"

def get_y(path):
    folders = glob(join(path, "*"))
    arrays = []
    for folder in folders:    
        array = np.genfromtxt(join(folder, f"average_train_{metric}.csv"), delimiter=",")
        arrays.append(array)
    return np.mean(arrays, axis=0)

xs = list()
ys = list()
for path in paths:
    y = get_y(path)
    ys.append(y)
    xs.append(np.arange(len(y)))
    

color = ["#1b9e77", "#e7298a", "#7570b3"]
labels = ["baseline", "p4", "hybrid"]

data = {"x": xs,
        "y": ys,
        "yl": None,
        "x_label": "Epoch", 
        "y_label": "bal-accuracy",
        "title": "",
        "save_path": f"/home/emcastro/{metric}.png",
        "color": color,
        "labels":labels}

plot(data)
