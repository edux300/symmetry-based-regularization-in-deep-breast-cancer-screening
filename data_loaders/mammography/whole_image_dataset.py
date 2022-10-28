#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 2022

@author: emcastro
"""
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from PIL import Image
from numpy import asarray
from os.path import join
import torch
import random
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from glob import glob
from os.path import basename

class WholeImageDatasetCMMD(Dataset):
    def __init__(self, directory, transform, n_copies=1):
        super(WholeImageDatasetCMMD, self).__init__()
        self.directory = directory
        self.transform = transform
        self.pats = self.get_all_pats()
        self.set_data()
        self.n_copies = n_copies
    
    def get_all_pats(self):
        files = glob(join(self.directory, "*.png"))
        pats = [basename(p).split("_")[0] for p in files]
        return list(set(pats))
    
    def set_data(self):
        self.image_files = []
        for pat in self.pats:
            for file in glob(join(self.directory, pat+"*.png")):
                self.image_files.append(file)
        self.class_weight = {0:0, 1:0}
        for file in self.image_files:
            self.class_weight[self.get_label(file)] += 1
        
        s = self.class_weight[0] + self.class_weight[1]
        self.class_weight[0] = s/(2 * self.class_weight[0])
        self.class_weight[1] = s/(2 * self.class_weight[1])

    
    def get_label(self, file):
        with open(file.replace(".png", ".txt")) as f:
            if f.readline().upper()=="MALIGNANT":
                return 1
            else:
                return 0

    def get_pat_level_label(self, pat):
        files = glob(join(self.directory, pat+"*.txt"))
        for file in files:
            with open(file) as f:
                if f.readline().upper()=="MALIGNANT":
                    return 1
        return 0

    def fold_split(self, fold, split="train"):
        #, val_size=0.15
        """
        Deterministic 5-fold cross validation
        Parameters
        ----------
        fold : 
            1 to 5 (default is 1)
        split :
            train | val | test
        """
        assert split in ["train", "test", "train_train", "train_val"]
        fold = int(fold)
        assert fold in [1,2,3,4,5]
        pats = sorted(self.pats)
        labels = [self.get_pat_level_label(pat) for pat in pats]
        test_splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        #pats, labels = test_splitter.split(pats, labels)
        
        
        if split == "test":
            inds = np.array([x for x in test_splitter.split(pats, labels)][fold-1][1])
            pats = [pats[i] for i in inds]
            labels = [labels[i] for i in inds]
            
        else:
            inds = np.array([x for x in test_splitter.split(pats, labels)][fold-1][0])
            pats = [pats[i] for i in inds]
            labels = [labels[i] for i in inds]

            if split == "train_train":
                val_splitter = StratifiedShuffleSplit(1,
                                                      test_size=.15,
                                                      random_state=43)
                inds = list(val_splitter.split(pats, labels))[0][0]
                pats = [pats[i] for i in inds]
                labels = [labels[i] for i in inds]

            elif split == "train_val":
                val_splitter = StratifiedShuffleSplit(1,
                                                      test_size=.15,
                                                      random_state=43)
                inds = list(val_splitter.split(pats, labels))[0][1]
                pats = [pats[i] for i in inds]
                labels = [labels[i] for i in inds]

            """
            val_splitter = StratifiedShuffleSplit(1, test_size=self.val_size,
                                                  random_state=43)

            if split == "valid":
                pats = list(val_splitter.split(pats, labels))[0][1]
            elif split == "train":
                pats = list(val_splitter.split(pats, labels))[0][0]
            """

        self.pats = pats
        print(pats)
        self.set_data()
        
    def load(self, file):
        image = asarray(Image.open(file))
        mask = np.load(file.replace(".png", "_M.npy"))
        return image, mask

    def __getitem__(self, idx):
        img, mask = self.load(self.image_files[idx])
        label = self.get_label(self.image_files[idx])
        if self.n_copies != 1:
            return [self.transform(img, mask) for _ in range(self.n_copies)], label
        return self.transform(img, mask), label

    def __len__(self):
        return len(self.image_files)

class WholeImageDatasetCBISSimple(WholeImageDatasetCMMD):
    def load(self, file):
        image = asarray(Image.open(file))
        return image
        
    def __getitem__(self, idx):
        img = self.load(self.image_files[idx])
        label = self.get_label(self.image_files[idx])
        if self.n_copies != 1:
            return [self.transform(img) for _ in range(self.n_copies)], label
        return self.transform(img), label

    def get_label(self, file):
        with open(file.replace(".png", ".txt")) as f:
            if f.readline().upper()=="MALIGNANT":
                return 1
            else:
                return 0

class WholeImageDatasetCBIS(WholeImageDatasetCMMD):
    def get_label(self, file):
        with open(file.replace(".png", "_case.txt")) as f:
            if f.readline().upper()=="MALIGNANT":
                return 1
            else:
                return 0

if __name__=="__main__":
    """
    print("Debugging CMMD")
    path = "/home/emcastro/datasets/CMMD"
    transform = lambda x:x
    dataset = WholeImageDataset(path, transform)
    """
    
    print("Debugging CBIS DDSM")
    path = "/home/emcastro/datasets/cbis_preprocessed4/val"
    transform = lambda x:x
    dataset = WholeImageDatasetCBIS(path, transform)