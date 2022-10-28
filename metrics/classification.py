#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:34:17 2020

@author: emcastro
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, f1_score, top_k_accuracy_score

def to_numpy(a):
    if not isinstance(a, np.ndarray):
        if isinstance(a, list):
            a = np.array(a)
        else:
            a = a.detach().cpu().numpy()
    return a

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def top5accuracy(out, y):
    out = to_numpy(out)
    y = to_numpy(y).reshape([-1, 1]).repeat(5, 1)
    predicted = np.argpartition(out, -5, axis=1)[:, -5:]
    return (predicted == y).sum()/predicted.shape[0]

def accuracy(out, y):
    out = to_numpy(out)
    y = to_numpy(y)
    if len(out.shape)>1:
        predicted = np.argmax(out, 1)
    else:
        predicted = (out>0.5)*1
    return (predicted == y).sum()/predicted.shape[0]

def bal_accuracy(out, y):
    out = to_numpy(out)
    y = to_numpy(y)
    if len(out.shape)>1:
        predicted = np.argmax(out, 1)
    else:
        predicted = (out>0.5)*1
    n_classes = np.max(y) + 1
    accuracies = []
    for c in range(n_classes):
        idx = y==c
        n = idx.sum()
        acc = (predicted[idx] == y[idx]).sum() / n
        accuracies.append(acc)
    return np.mean(accuracies)

def rocauc(out, y, negative_class=None, weighted=False):
    average = "macro" if not weighted else "weighted"
    out = to_numpy(out)
    y = to_numpy(y)
    if negative_class is None:
        if len(out.shape) == 1:
            return roc_auc_score(y, out)
        if out.shape[1]==2:
            return roc_auc_score(y, softmax(out)[:,1], average=average, multi_class='ovo')
        else:
            return roc_auc_score(y, softmax(out), average=average, multi_class='ovo')
    else:
        negatives = y==negative_class
        positive_classes = np.unique(y[np.logical_not(negatives)])
        scores = []
        for c in positive_classes:
            local_out = np.concatenate([out[y==c, c], out[negatives, c]])
            local_y   = np.concatenate([y[y==c]//c, y[negatives]])
            scores.append(roc_auc_score(local_y, local_out))
        return np.mean(scores)

def f1score(out, y, negative_class=None, weighted=False):
    average = "macro" if not weighted else "weighted"
    out = to_numpy(out)
    y = to_numpy(y)
    if negative_class is None:
        if len(out.shape)>1:
            predicted = np.argmax(out, 1)
        else:
            predicted = (out>0.5)*1
        return f1_score(y, predicted, average=average)
    else:
        negatives = y==negative_class
        positive_classes = np.unique(y[np.logical_not(negatives)])
        scores = []
        for c in positive_classes:
            local_out = np.concatenate([out[y==c, c], out[negatives, c]])
            local_y   = np.concatenate([y[y==c]//c, y[negatives]])
            scores.append(f1_score(local_y, local_out>0.5))
        return np.mean(scores)
            

def cross_entropy(out, y):
    out = to_numpy(out)
    y = to_numpy(y)
    return log_loss(y, softmax(out))
