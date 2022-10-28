"""
Engine description:
    A "data_loader" is iterated over all its "batch"es and for each batch
    it will run a "train_fn". Some code will be run at the start of the epoch,
    the end of the epoch and after each "train_fn" call.
"""

class Callback():
    def on_epoch_started(self, resources):
        pass
    def on_epoch_ended(self, resources):
        pass
    def on_batch_started(self, resources):
        pass
    def on_batch_ended(self, resources):
        pass
    def on_train_started(self, resources):
        pass
    def on_train_ended(self, resources):
        pass
    def on_valid_started(self, resources):
        pass
    def on_valid_ended(self, resources):
        pass

class MainCallback(Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.on_epoch_started = self.run("on_epoch_started")
        self.on_epoch_ended = self.run("on_epoch_ended")
        self.on_batch_started = self.run("on_batch_started")
        self.on_batch_ended = self.run("on_batch_ended")
        self.on_train_started = self.run("on_train_started")
        self.on_train_ended = self.run("on_train_ended")
        self.on_valid_started = self.run("on_valid_started")
        self.on_valid_ended = self.run("on_valid_ended")

    def run(self, fn_name):
        def all_runs(resources):
            for c in self.callbacks:
                getattr(c, fn_name)(resources)
        return all_runs

class FunctionCallback(Callback):
    def __init__(self, k, func):
        self.k = k
        self.func = func
    def on_epoch_started(self, resources):
        if self.k == "on_epoch_started":
            self.func(resources)
    def on_epoch_ended(self, resources):
        if self.k == "on_epoch_ended":
            self.func(resources)
    def on_batch_started(self, resources):
        if self.k == "on_batch_started":
            self.func(resources)
    def on_batch_ended(self, resources):
        if self.k == "on_batch_ended":
            self.func(resources)
    def on_train_started(self, resources):
        if self.k == "on_train_started":
            self.func(resources)
    def on_train_ended(self, resources):
        if self.k == "on_train_ended":
            self.func(resources)
    def on_valid_started(self, resources):
        if self.k == "on_valid_started":
            self.func(resources)
    def on_valid_ended(self, resources):
        if self.k == "on_valid_ended":
            self.func(resources)


def train_epoch(epoch, train_fn, valid_fn, resources, callbacks, max_iterations=None):
    resources["epoch"] = epoch
    callbacks.on_epoch_started(resources)

    callbacks.on_train_started(resources)
    for i, data in enumerate(resources["train_data_loader"]):
        resources["batch"] = data
        callbacks.on_batch_started(resources)
        train_fn(resources)
        callbacks.on_batch_ended(resources)
        if max_iterations is not None:
            if i >=  max_iterations:
                break
    callbacks.on_train_ended(resources)

    if resources["val_data_loader"] is not None:
        callbacks.on_valid_started(resources)
        for data in resources["val_data_loader"]:
            resources["batch"] = data
            callbacks.on_batch_started(resources)
            valid_fn(resources)
            callbacks.on_batch_ended(resources)
        callbacks.on_valid_ended(resources)

    callbacks.on_epoch_ended(resources)

def train(epochs, train_fn, valid_fn, resources, callbacks, max_iterations=None):
    validate_at_the_end = False
    if max_iterations is None:
        import math
        max_iterations = math.inf
        validate_at_the_end = True

    resources["epoch"] = 0
    resources["engine_finish_flag"] = False
    counter = 0

    def _validate():
        nonlocal counter
        callbacks.on_train_ended(resources)
        callbacks.on_valid_started(resources)
        for data in resources["val_data_loader"]:
            resources["batch"] = data
            callbacks.on_batch_started(resources)
            valid_fn(resources)
            callbacks.on_batch_ended(resources)
        callbacks.on_valid_ended(resources)
        callbacks.on_epoch_ended(resources)
        counter = 0
        resources["epoch"] += 1

    while True:
        for i, data in enumerate(resources["train_data_loader"]):
            if resources["epoch"] == epochs or resources["engine_finish_flag"]:
                return

            if counter == 0:
                callbacks.on_epoch_started(resources)
                callbacks.on_train_started(resources)
            counter += 1
            resources["batch"] = data
            callbacks.on_batch_started(resources)
            train_fn(resources)
            callbacks.on_batch_ended(resources)

            if counter >= max_iterations:
                _validate()

        if validate_at_the_end:
            counter=0
            _validate()
