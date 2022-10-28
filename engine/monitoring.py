from engine.main import Callback
import numpy as np
import torch
import pickle as pkl

class WeightNormMonitoring(Callback):
    def __init__(self, model, path, steps_per_measure=20):
        self.model = model
        self.parameters_norms = {n:[] for n, _ in model.named_parameters()}
        self.steps_per_measure = steps_per_measure
        self.steps=0
        self.path = path

    def on_batch_started(self, resources):
        if self.steps%self.steps_per_measure==0:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    self.parameters_norms[n].append(float(torch.norm(p).cpu().numpy()))
        self.steps+=1

    def on_epoch_ended(self, resources):
        self.save()

    def save(self):
        with open(self.path+"/weight_norms.pkl", "wb") as file:
            pkl.dump(self.parameters_norms, file)

