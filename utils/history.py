import pickle as pkl
import numpy as np
import utils.metrics as metrics


class History():
    """
    Auxiliary class to record some values during training
    """
    def __init__(self):
        self.d = dict()
        self.e = 0
        self.state = None

    def epoch(self, e):
        self.e = e

    def addm(self, keys, values):
        for k, v in zip(keys, values):
            self.add(k, v)
    
    def add(self, k, v):
        if not k in self.d:
            self.d[k] = []
        self.d[k].append((self.e, v))

    def get_epoch_average(self, k):
        tuples = self.d[k]
        epochs = set()
        values_per_epoch = dict()
        for e, v in tuples:
            epochs.add(e)
            if e not in values_per_epoch:
                values_per_epoch[e] = []
            values_per_epoch[e].append(v)
        epochs = list(epochs)
        epochs.sort()
        values = []
        for e in epochs:
            values.append(np.mean(values_per_epoch[e]))
        return epochs, values
    
    def get_curr_epoch_average(self, k):
        tuples = self.d[k]
        values = []
        for e, v in tuples:
            if e==self.e:
                values.append(v)
        return np.mean(values)
    
    def get_plottables(self, keys):
        xxs, yys, legs = list(), list(), list()
        for k in keys:
            xx, yy = self.get_epoch_average(k)
            xxs.append(xx)
            yys.append(yy)
            legs.append(k)
        return xxs, yys, legs

    def save(self, path):
        with open(path, "wb") as file:
            pkl.dump((self.d, self.e, self.state), file)
    
    def load(self, path):
        with open(path, "rb") as file:
            self.d, self.e, self.state = pkl.load(file)


class History_Classification(History):
    def __init__(self):
        super(History_Classification, self).__init__()
        self._empty_trackers()
    
    def _empty_trackers(self):
        self.out = []
        self.y = []
    
    def __call__(self, out, y):
        self.out.append(metrics.to_numpy(out))
        self.y.append(metrics.to_numpy(y))
        
    def process_train(self):
        out = np.concatenate(self.out)
        y = np.concatenate(self.y)
        self.add("train_AUC", metrics.rocauc(out, y))
        self.add("train_Acc", metrics.accuracy(out, y))
        self.add("train_loss", metrics.cross_entropy(out, y))
        self._empty_trackers()
    
    def process_val(self):
        out = np.concatenate(self.out)
        y = np.concatenate(self.y)
        self.add("val_AUC", metrics.rocauc(out, y))
        self.add("val_Acc", metrics.accuracy(out, y))
        self.add("val_loss", metrics.cross_entropy(out, y))
        self._empty_trackers()    

    @property
    def auc(self):
        return self.get_curr_epoch_average("val_AUC")
    
    @property
    def whole_img_auc(self):
        return self.get_curr_epoch_average("Whole Image AUC")

    @property
    def acc(self):
        return self.get_curr_epoch_average("val_Acc")

    def __str__(self):
        values = [self.get_curr_epoch_average("train_loss"),
                  self.get_curr_epoch_average("train_Acc"),
                  self.get_curr_epoch_average("train_AUC")]

        s = "Train\n\tCross-Entropy: {:3f}\n\tAccuracy: {:3f}\n\tAUC: {:3f}".format(*values)
        
        values = [self.get_curr_epoch_average("val_loss"),
                  self.get_curr_epoch_average("val_Acc"),
                  self.get_curr_epoch_average("val_AUC")]
        
        s+= "\nValidation\n\tCross-Entropy: {:3f}\n\tAccuracy: {:3f}\n\tAUC: {:3f}".format(*values)
        return s
        
        
        
        
        