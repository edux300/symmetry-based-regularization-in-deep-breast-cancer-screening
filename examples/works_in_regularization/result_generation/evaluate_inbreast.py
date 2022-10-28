from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
from os.path import join, basename
import numpy as np
import pickle as pkl
import sys
import yaml
import os
import pandas as pd
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')

from models.classification import get_model
from metrics.classification import accuracy, bal_accuracy, rocauc, f1score
from data_loaders.data_augmentation import get_transform
from data_loaders.mammography.sampling_dataset import PresampledImageDataset

prepared = False
def prepare(base_path):
    global test_loader, prepared
    basic_config = get_config(base_path)
    basic_config["test_get_transform"]["mean"] += 0.101
    basic_config["test_get_transform"]["std"] /= 0.750/0.617
    val_transform = get_transform(**basic_config["test_get_transform"])

    # use validation set configurations except path
    # changed
    basic_config["val_dataset"]["directory"] = "/home/emcastro/datasets/inbreast_test/whole_test"
    #basic_config["val_dataset"]["directory"] = "/home/emcastro/datasets/cbis_preprocessed4/test"
    test_dataset = PresampledImageDataset(**basic_config["val_dataset"], transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=basic_config["training"]["batch_size"], shuffle=False, num_workers=4)
    prepared = True

def evaluate(path, metric, metric_func):
    global test_loader
    config = get_config(path)
    device = torch.device(config["training"]["device"])
    model = get_model(**config["model"])

    weights_path = glob(join(path, f"model_average_valid_{metric}_*.pth"))
    weights_path.sort(key=lambda x:float(x.split(".")[-2].split("_")[-1]), reverse=True)
    best_weights = weights_path[0]

    model.load_state_dict(torch.load(best_weights))
    model.to(device)
    model.eval()

    probs = []
    labels = []
    for X, y in test_loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            out = model(X)

        probs.append(out.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    value = metric_func(probs, labels)

    return value

def get_config(path):
    with open(join(path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    from metrics.classification import accuracy, bal_accuracy, rocauc, f1score

    metric_funcs = {"accuracy": accuracy,
                    "bal-accuracy": bal_accuracy,
                    "rocAUC": rocauc,
                    "f1-score": f1score,}

    print("Evaluating models:")
    for i, path in enumerate(sys.argv[1:]):
        print(f"\t{i+1}. {path}")

    lines = []
    lines.append("Start\n")
    lines.append("models:\n")
    for path in sys.argv[1:]:
        lines.append(path+"\n")

    lines.append("metrics:\n")
    for k, f in metric_funcs.items():
        values = []
        for path in sys.argv[1:]:
            prepare(path)
            value = evaluate(path, k, f)
            values.append(value)
            print(value)

        lines.append(f"{k}, {np.mean(values)}, {np.std(values)}\n")

    lines.append("Finish\n")
    with open("results.txt", "a") as file:
        file.writelines(lines)
