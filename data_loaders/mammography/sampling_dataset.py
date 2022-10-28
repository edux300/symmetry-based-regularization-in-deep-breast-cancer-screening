#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:03:50 2020

@author: emcastro
"""
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from PIL import Image
from numpy import asarray
from os.path import join, basename
import torch
import random
from torch.utils.data._utils.collate import default_collate

class PresampledImageDataset(Dataset):
    def __init__(self, directory, preload=False, half_size=112, transform=None,
                 admitted_classes=None, n_copies=1, artificial_enlarge=False,
                 join_by_malignancy=False):
        super(PresampledImageDataset, self).__init__()
        self.directory = directory
        self.half_size = half_size
        self.transform = transform
        self.n_copies = n_copies

        with open(join(directory, "sampled_points.pkl"), "rb") as file:
            self.sampled_coords = pkl.load(file)

        touched = set()
        self.sampled_filtered = []
        for x in self.sampled_coords:
            if x[2] == "BACKGROUND":
                if x[0] not in touched:
                    touched.add(x[0])
                    self.sampled_filtered.append(x)
            else:
                self.sampled_filtered.append(x)
        self.sampled_coords = self.sampled_filtered

        if admitted_classes is not None:
            self.sampled_coords = [x for x in self.sampled_coords if x[2] in admitted_classes]

        if artificial_enlarge:
            self.sampled_coords_larger = []
            ts = []
            def get_t(dx, dy): return lambda coords: (coords[0]-dx*half_size, coords[1]-dy*half_size)
            for dx in np.linspace(-0.05,0.05,3):
                for dy in np.linspace(-0.05,0.05,3):
                    ts.append(get_t(dx, dy))

            for x in self.sampled_coords:
                samples = [[x[0], t(x[1]), x[2]] for t in ts]
                self.sampled_coords_larger.extend(samples)
            self.sampled_coords = self.sampled_coords_larger

        if preload:
            self.preloaded = []
            for file, point, y in self.sampled_coords:
                self.preloaded.append(self.load(file, point), y)
        else:
            self.preloaded = None

        if not join_by_malignancy:
            self.t_label = lambda x:x
        else:
            self.t_label = lambda x:x.split("_")[0]

        self.ys = [self.t_label(y) for _,_, y in self.sampled_coords]
        classes = sorted(list(set(self.ys)))
        self.class_dict = dict()
        for i, c in enumerate(classes):
            self.class_dict[c] = i
        self.class_weight = {c: self.ys.count(c) for c in classes}

    def __getitem__(self, idx):
        if self.preloaded is None:
            file, point, y = self.sampled_coords[idx]
            patch = self.load(file, point)

        else:
            _, _, y = self.sampled_coords[idx]
            patch = self.preloaded[idx]

        if self.n_copies != 1:
            return [self.transform(patch) for _ in range(self.n_copies)], self.to_numeric(y)
        else:
            if self.transform:
                patch = self.transform(patch)

            return patch, self.to_numeric(y)

    def __len__(self):
        return len(self.sampled_coords)

    def load(self, file, point, safe_padding=50):
        #print(join(self.directory, file))
        #print(self.directory, file, join(self.directory, file))
        image = asarray(Image.open(join(self.directory, basename(file))))
        x, y = point
        image = np.pad(image, self.half_size+safe_padding, "constant")
        x = round(x)
        y = round(y)
        return image[x+safe_padding:x+self.half_size*2+safe_padding,
                     y+safe_padding:y+self.half_size*2+safe_padding]

    def to_numeric(self, y):
        return self.class_dict[self.t_label(y)]

    def sample_random_batch(self, size):
        if not hasattr(self, "_idx_range"):
            self._idx_range = range(len(self))
        idxs = random.sample(self._idx_range, size)
        return default_collate([self[idx] for idx in idxs])

    def sample_from_class(self, class_list):
        if not hasattr(self, "_per_class_idx_range"):
            self._per_class_idx_range = {v: [i for i, x in enumerate(self.sampled_coords) if x[2] == c]
                                             for c, v in self.class_dict.items()}
        idxs = []
        for c in class_list:
            idxs.append(random.sample(self._per_class_idx_range[c], 1)[0])
        return default_collate([self[idx] for idx in idxs])
