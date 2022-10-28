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

#from models.classification import get_model
from models.classification.init_equivariant import get_model
from metrics.classification import accuracy, bal_accuracy, rocauc, f1score
#from data_loaders.data_augmentation import large_whole_image_tranform
from data_loaders.data_augmentation import simple_large_whole_image_tranform
from data_loaders.mammography.sampling_dataset import PresampledImageDataset

from examples.works_in_regularization.train_whole2 import WholeImageModel
from data_loaders.mammography.whole_image_dataset import WholeImageDatasetCBIS, WholeImageDatasetCMMD, WholeImageDatasetCBISSimple

prepared = False

dataset_name = "CBIS"
def get_config(path):
    with open(join(path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config

def prepare(base_path):
    global test_loader, prepared, test_dataset
    basic_config = get_config(base_path)
    #val_transform = simple_large_whole_image_tranform(**basic_config["test_get_transform"])
    

    if dataset_name == "CBIS":
        val_transform = simple_large_whole_image_tranform(**basic_config["test_get_transform"])
        basic_config["dataset"]["directory"] = os.environ["DEEPMM_DATADIR"] + "/cbis_whole_image_no_repeat_adjust/test/"
        #basic_config["dataset"]["directory"] = os.environ["DEEPMM_DATADIR"] + "/cbis_whole_image_no_repeat/test/"
        
        test_dataset = WholeImageDatasetCBISSimple(**basic_config["dataset"], transform=val_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=basic_config["training"]["batch_size"], shuffle=False, num_workers=4)
        prepared = True
    elif dataset_name == "CMMD":
        val_transform = simple_large_whole_image_tranform(**basic_config["test_get_transform"])
        basic_config["dataset"]["directory"] = os.environ["DEEPMM_DATADIR"] + "/CMMD_classification/"        
        test_dataset = WholeImageDatasetCBISSimple(**basic_config["dataset"],
                                                   transform=val_transform)
        test_dataset.fold_split(basic_config["training"]["fold"], "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=basic_config["training"]["batch_size"], shuffle=False, num_workers=4)
        prepared = True
    elif dataset_name == "INbreast":
        val_transform = simple_large_whole_image_tranform(**basic_config["test_get_transform"])
        test_dataset = WholeImageDatasetCBISSimple(**basic_config["dataset"], transform=val_transform)
        test_dataset.fold_split(basic_config["training"]["fold"], "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=basic_config["training"]["batch_size"], shuffle=False, num_workers=4)
        prepared = True


def evaluate(path, metric, metric_func):
    global test_loader, test_dataset
    config = get_config(path)
    device = torch.device(config["training"]["device"])
    
    # model
    backbone = get_model(**config["model"], group_equiv_expand=False)
    with torch.no_grad():
        _, in_channels, H, W = backbone.features(torch.rand(2, 1, *config["test_get_transform"]["size"])).shape
    k = int(H*W*0.5)    
    model = WholeImageModel(backbone, in_channels, k=k, pooling_type=config["training"]["pooling_type"])
    model = model.to(device)
    #print(model)

    weights_path = glob(join(path, f"model_average_valid_{metric}_*.pth"))
    weights_path.sort(key=lambda x:float(x.split(".")[-2].split("_")[-1]), reverse=True)
    
    last_model_weights = True
    if last_model_weights:
        weights_path = glob(join(path, "model.pth"))
    print(f"Last model weights (should be True for INbreast only): {last_model_weights}")
    print(weights_path)
    best_weights = weights_path[0]

    model.load_state_dict(torch.load(best_weights))
    model.to(device)
    
    # tune bn
    """
    model.train()
    for X, y in test_loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            _ = model(X)
    """
    
    model.eval()
    probs = []
    labels = []
    for X, y in test_loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            out = model(X)

        probs.append(torch.sigmoid(out).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    value = metric_func(probs, labels)
    
    """
    print(probs>0.5)
    print(labels)
    for i, n in enumerate(test_dataset.image_files):
        if ((probs>0.5) != labels)[i]:
            print(n)
    """

    return probs, labels, value

if __name__ == "__main__":
    metric_funcs = {"accuracy":accuracy,
                    "rocAUC": rocauc}
    
    evaluate_ensemble = False
    join_all_preds = True
    
    print(f"Join all preds (5-fold): {join_all_preds}")
    print(f"Dataset name: {dataset_name}")
    
    
    print("Evaluating models:")
    for i, path in enumerate(sys.argv[1:]):
        print(f"\t{i+1}. {path}")

    lines = []
    lines.append("Start\n")
    lines.append("models:\n")
    for path in sys.argv[1:]:
        lines.append(path+"\n")

    lines.append("metrics:\n")
    small_stats = f"{sys.argv[1]}"

    for k, f in metric_funcs.items():
        values = []
        predictions_all = []
        labels_all = []
        for path in sys.argv[1:]:
            prepare(path)
            probs, labels, value = evaluate(path, k, f)
            print(probs)
            print(labels)
            values.append(value)
            print(value)
            predictions_all.append(probs)
            labels_all.append(labels)
        
        if evaluate_ensemble:
            assert all([np.all(x==labels_all[0]) for x in labels_all])
            predictions_all = np.stack(predictions_all, 0).mean(0)
            ensemble_result = f(predictions_all, labels_all[0])
        
        
            with open("ensemble_result.txt", "a") as file:
                file.writelines(str(ensemble_result) + "\n")
            
        if join_all_preds:
            predictions_all = np.concatenate(predictions_all)
            labels_all = np.concatenate(labels_all)
            fvalue = f(predictions_all, labels_all)
            lines.append(f"{k}, {fvalue }, 0, {fvalue }\n")
            small_stats += f", {fvalue }"
            small_stats += f", {0}"

        else:
            lines.append(f"{k}, {np.mean(values)}, {np.std(values)}, {values}\n")
            small_stats += f", {np.mean(values)}"
            small_stats += f", {np.std(values)}"
    small_stats+="\n"
    lines.append("Finish\n")
    with open("results.txt", "a") as file:
        file.writelines(lines)
    
    with open("results_small.txt", "a") as file:
        file.writelines(small_stats)
    


