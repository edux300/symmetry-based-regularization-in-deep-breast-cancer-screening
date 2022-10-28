from engine.main import Callback
import numpy as np
class WarmupScheduler(Callback):
    def __init__(self, optimizer, learning_rate, warmup_iters):
        self.optimizer=optimizer
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.curr = 0

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def on_batch_started(self, resources):

        if self.curr==-1:
            return

        if self.curr < self.warmup_iters:
            self.curr+=1
            lr = self.learning_rate * (self.curr / self.warmup_iters)
            self.set_lr(lr)

        elif self.curr==self.warmup_iters:
            print(f"Finished warmup, current learning rate {self.learning_rate}")
            self.curr=-1
            self.set_lr(self.learning_rate)

class FixedDropScheduler(Callback):
    def __init__(self, optimizer, learning_rate, reduce_epochs=[], reduce_factor=10):
        self.optimizer=optimizer
        self.learning_rate = learning_rate
        self.reduce_epochs = reduce_epochs
        self.reduce_factor = reduce_factor

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def lower_lr(self):
        self.learning_rate /= self.reduce_factor
        print(f"Reducing, learning rate {self.learning_rate}")
        self.set_lr(self.learning_rate)

    def on_epoch_started(self, resources):
        if resources["epoch"] in self.reduce_epochs:
            self.lower_lr()

class EarlyStopping(Callback):
    def __init__(self, max_patience=10, key="", mode="max"):
        assert mode in ["max", "min"]
        self.max_patience = max_patience
        self.key = key
        self.mode = mode
        self.patience = 0
        self.curr = -np.inf if mode=="max" else np.inf

    def on_epoch_ended(self, resources):
        if self.mode == "max":
            new_best = self.curr < resources[self.key]
        else:
            new_best = self.curr > resources[self.key]

        if new_best:
            self.curr = resources[self.key]
            self.patience = 0
        else:
            self.patience+=1

        if self.patience>self.max_patience:
            resources["engine_finish_flag"] = True

class DynamicDropScheduler(Callback):
    def __init__(self, optimizer, learning_rate, key="", mode="max",
                 reduce_factor=10, max_patience=10, min_epoch=0, tol=0.0,
                 set_finish_flag_after=None):
        self.optimizer=optimizer
        self.learning_rate = learning_rate
        self.key=key
        if type(key) in [list, tuple]:
            self.curr = [-np.inf if mode=="max" else np.inf for k in key]
            self.multiple = True
        else:
            self.curr=-np.inf if mode=="max" else np.inf
            self.multiple = False
        self.tol = tol
        if mode=="min":
            raise(NotImplementedError())

        self.mode=mode
        self.reduce_factor = reduce_factor
        self.max_patience = max_patience
        self.min_epoch = min_epoch - 1 # start at the end of (min_epoch-1)
        self.patience = 0
        self.num_drops = 0
        self.set_finish_flag_after = set_finish_flag_after

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def lower_lr(self):
        self.learning_rate /= self.reduce_factor
        print(f"Reducing, learning rate {self.learning_rate}")
        self.set_lr(self.learning_rate)

    def on_epoch_ended(self, resources):
        e = resources["epoch"]

        if e < self.min_epoch:
            return

        if self.multiple:
            for i, (key, curr) in enumerate(zip(self.key, self.curr)):
                if curr + self.tol < resources[key]:
                    self.curr[i]=resources[self.key[i]]
                    self.patience=-1
        else:
            if self.curr + self.tol < resources[self.key]:
                self.curr=resources[self.key]
                self.patience=-1

        self.patience+=1

        if self.patience>self.max_patience:
            self.patience=0
            self.lower_lr()
            self.num_drops+=1
        if self.set_finish_flag_after is not None:
            if self.num_drops == self.set_finish_flag_after:
                resources["engine_finish_flag"] = True

