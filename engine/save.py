from engine.main import Callback
import torch
import numpy as np
import os


class saveLastKModel(Callback):
    def __init__(self, path, model, k):
        self.path = path
        self.model = model
        self.k = k
        self.saved_paths = []

    def on_epoch_ended(self, resources):
        k = resources["epoch"]
        self.saved_paths.append(self.path.replace(".pth", f"_{k}.pth"))
        torch.save(self.model.state_dict(), self.path.replace(".pth", f"_{k}.pth"))
        if len(self.saved_paths) > self.k:
            path = self.saved_paths[0]
            del self.saved_paths[0]
            os.remove(path)

class saveBestKModel(Callback):
    def __init__(self, path, model, k, metric_key, mode="max"):
        self.path = path
        self.model = model
        self.k = k
        self.saved_paths = []
        self.scores = []
        self.curr = -np.inf if mode == "max" else np.inf
        self.patience = 0
        self.mode = mode
        self.metric_key = metric_key

    def on_epoch_ended(self, resources):
        value = resources[self.metric_key]
        if value > self.curr if self.mode=="max" else value < self.curr:
            epoch = resources["epoch"]
            self.scores.append(value)
            str_value = "{:.6f}".format(value)
            path = self.path.replace(".pth", f"_{epoch}_{str_value}.pth")
            self.saved_paths.append(path)
            torch.save(self.model.state_dict(), path)
            if len(self.scores) > self.k:
                idx = np.argmin(self.scores) if self.mode == "max" else np.argmax(self.scores)
                os.remove(self.saved_paths[idx])
                del self.scores[idx]
                del self.saved_paths[idx]
                self.curr = np.min(self.scores) if self.mode == "max" else np.max(self.scores)
                self.patience = 0
        else:
            if len(self.scores) == self.k:
                self.patience += 1     

class saveMultiBestKModel(Callback):
    def __init__(self, path, model, k, metric_keys, modes="max"):
        if isinstance(modes, str):
            modes = [modes]*len(metric_keys)
        paths = [path.replace(".pth", f"_{m}.pth") for m in metric_keys]
        self.savers = []
        for path, m, mode in zip(paths, metric_keys, modes):
            self.savers.append(saveBestKModel(path, model, k, m, mode))
        self.patience = 0

    def on_epoch_ended(self, resources):
        ps = []
        for s in self.savers:
            s.on_epoch_ended(resources)
            ps.append(s.patience)
        self.patience = min(ps)
    
    def reset_patience(self):
        self.patience=0
        for s in self.savers:
            s.patience=0

class saveBestModelCallback(Callback):
    def __init__(self, path, model, metric_key, mode="max"):
        assert mode in ["max", "min"]
        self.curr = -np.inf if mode == "max" else np.inf
        self.path = path
        self.model = model
        self.metric_key = metric_key
        self.mode = mode

    def on_epoch_ended(self, resources):
        value = resources[self.metric_key]
        if value > self.curr if self.mode=="max" else value < self.curr:
            self.curr = value
            torch.save(self.model.state_dict(), self.path)
            print("\tModel Saved")

class saveLastModelCallback(Callback):
    def __init__(self, path, model):        
        self.path = path
        self.model = model

    def on_epoch_ended(self, resources):
        torch.save(self.model.state_dict(), self.path)
        print("\tModel Saved")

class saveAllModelCallback(Callback):
    def __init__(self, path, model):
        self.path = path
        self.model = model

    def on_epoch_ended(self, resources):
        path = self.path.replace(".pth", f"_{resources['epoch']}.pth")
        torch.save(self.model.state_dict(), path)
        