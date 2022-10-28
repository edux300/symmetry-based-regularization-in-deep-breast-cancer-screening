#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:24:02 2020

@author: emcastro
"""
from engine.main import Callback
import time
import os
from os.path import join, isfile

class Logger(Callback):
    def __init__(self, keys, names=None, print_to_file=None):
        self.epoch = 0
        self.keys = keys
        self.names = names if names else keys
        self.train_time_taken = 0
        self.valid_time_taken = 0
        self.print_to_file = print_to_file

    def on_epoch_started(self, resources):
        self.epoch = resources["epoch"]

    def on_epoch_ended(self, resources):
        string = "\nEpoch {}".format(self.epoch)
        # profiler
        string += "\n================="
        string += "\nTime Profile:"
        string += "\n\tTrain: {:.1f}s".format(self.train_time_taken)
        string += "\n\tValidation: {:.1f}s".format(self.valid_time_taken)
        string += "\n================="

        # Keys to print in the log
        for k, name in zip(self.keys, self.names):
            string += "\n\t{}: {}".format(name, resources[k])

        print(string)
        if self.print_to_file is not None:
            with open(self.print_to_file, "a") as file:
                file.write(string)

    def on_batch_ended(self, resources):
        self.counter += 1
        print("\rCurrent Progress ({}): {:.1f}%\t\t\t".format(self.curr_loop, self.counter/self.curr_data_loader_len * 100), end="")

    def on_train_started(self, resources):
        self.train_start_time = time.time()
        self.curr_data_loader_len = len(resources["train_data_loader"])
        self.curr_loop = "train"
        self.counter = 0

    def on_train_ended(self, resources):
        self.train_time_taken = time.time()-self.train_start_time

    def on_valid_started(self, resources):
        self.valid_start_time = time.time()
        self.curr_data_loader_len = len(resources["val_data_loader"])
        self.curr_loop = "validation"
        self.counter = 0

    def on_valid_ended(self, resources):
        self.valid_time_taken = time.time()-self.valid_start_time

class FileBatchLogger(Callback):
    def __init__(self, path, keys, files=None, delete_if_exists=False, phase="train"):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.keys = keys
        self.files = []
        files = files if files else ["{}.csv".format(k) for k in keys]
        for f in files:
            name = join(path, f)
            self.files.append(name)
            if isfile(name) and delete_if_exists:
                os.remove(name)
        self.phase = phase
    def on_train_started(self, resources):
        self.active=False
        if self.phase in ["train", "both"]:
            self.active=True

    def on_valid_started(self, resources):
        self.active=False
        if self.phase in ["valid", "both"]:
            self.active=True

    def on_batch_ended(self, resources):
        if not self.active:
            return
        for k, f in zip(self.keys, self.files):
            if not isfile(f):
                with open(f, "w") as file:
                    file.write("{}".format(resources[k]))
            else:
                with open(f, "a") as file:
                    file.write(", {}".format(resources[k]))

class FileEpochLogger(Callback):
    def __init__(self, path, keys, files=None, delete_if_exists=False):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.keys = keys
        self.files = []
        files = files if files else ["{}.csv".format(k) for k in keys]
        for f in files:
            name = join(path, f)
            self.files.append(name)
            if isfile(name) and delete_if_exists:
                os.remove(name)

    def on_epoch_ended(self, resources):
        for k, f in zip(self.keys, self.files):
            if not isfile(f):
                with open(f, "w") as file:
                    file.write("{}".format(resources[k]))
            else:
                with open(f, "a") as file:
                    file.write(", {}".format(resources[k]))