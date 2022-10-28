import random
import numpy as np
import argparse
from os.path import join
import os
import json

"""
Constants
========================================
"""
RESULTS_DIR = os.environ["DEEPMM_RESULTSDIR"] + "/Works_on_Regularization/"
CBIS_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/cbis_preprocessed4/"
CUSTOM_PRELOAD_PATH = os.environ["DEEPMM_RESULTSDIR"] + "/pretrain_imagenet/"
DEVICE = "cuda:0"

"""
python3 examples/works_in_regularization/train.py --arquitecture p4_64 --pretrained custom_20 --optimization scratch_AdamW --data CBIS_mass --augmentation base;
"""

def argparse_config():
    argparser = argparse.ArgumentParser(description='Deep Learning Regularization Parameters')
    #argparser.add_argument('--arquitecture', type=str, default="z2_64", help='<group_equiv>_<width>')
    #argparser.add_argument('--optimization',  type=str, default="scratch_SGD", help='<mode>_<optimizer>')
    argparser.add_argument('--architecture', type=str, default="convnext_tiny", help='options: vgg16, resnet34, resnet50, densenet121, convnext_tiny, efficientnet_b4')
    argparser.add_argument('--group_equiv_type', type=str, default="p4")
    argparser.add_argument('--n_group_blocks', type=int, default=0, help='number_of_equivariant_blocks')
    #argparser.add_argument('--width_multiplier', type=float, default=1, help='network width multiplier')
    #argparser.add_argument('--optimizer', type=str, default="AdamW", help='SGD, Adam, AdamW')
    #argparser.add_argument('--learning_rate', type=float, default=0.000125, help='effective batch size (accumulation)')
    #argparser.add_argument('--optimizer', type=str, default="Adam", help='SGD, Adam, AdamW')
    #argparser.add_argument('--learning_rate', type=float, default=0.000125, help='effective batch size (accumulation)')
    argparser.add_argument('--optimizer', type=str, default="SGD", help='SGD, Adam, AdamW')
    argparser.add_argument('--learning_rate', type=float, default=0.01, help='effective batch size (accumulation)')
    #argparser.add_argument('--optimizer_kw_params', type=json.loads, default='{"weight_decay":0.0005}')
    argparser.add_argument('--optimizer_kw_params', type=json.loads, default='{"momentum":0.8, "nesterov": true, "weight_decay":0.0001}')
    argparser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    argparser.add_argument('--warmup_iter', type=int, default=1000, help='number of warmup_iterations')
    argparser.add_argument('--drop_epochs', nargs='+', type=int, default=[1500, 1750], help='number of epochs')
    argparser.add_argument('--real_bs', type=int, default=32, help='batch size')
    argparser.add_argument('--equiv_bs', type=int, default=128, help='effective batch size (accumulation)')

    argparser.add_argument('--pretrained', type=str, default="None", help='supported: None, default, custom_<epoch>')
    argparser.add_argument('--data', type=str,   default='CBIS_mass', help='supported: CBIS')
    argparser.add_argument('--contrastive', default=False, action='store_true')
    argparser.add_argument('--contrastive_lambda', type=float, default=1.0)
    argparser.add_argument('--contrastive_n_copies', type=int, default=4)
    argparser.add_argument('--augmentation', default='base', type=str, help='supported: none, base, all, flips, rotation,\
                                                             translation, intensity, scale, elastic')
    argparser.add_argument('--save_path', default='resnet50_malignancy', type=str, help='allows multiple runs')
    return argparser.parse_args()

def get_config(parse=True):
    args = argparse_config()

    dataset, target = args.data.split("_")
    contrastive = args.contrastive
    augmentation = args.augmentation

    assert args.optimizer in ["SGD", "Adam", "AdamW"]

    join_by_malignancy = False
    if target=="mass":
        admitted_classes = ["BACKGROUND", "BENIGN_MASS", "MALIGNANT_MASS"]
    elif target=="massonly":
        admitted_classes = ["BENIGN_MASS", "MALIGNANT_MASS"]
    elif target=="calc":
        admitted_classes = ["BACKGROUND", "BENIGN_CALC", "MALIGNANT_CALC"]
    elif target=="malignancy":
        admitted_classes = ["BACKGROUND", "BENIGN_CALC", "BENIGN_MASS", "MALIGNANT_CALC", "MALIGNANT_MASS"]
        join_by_malignancy = True
    else:
        raise(ValueError("Unknown target argument."))

    # training config:
    logs_path = join(RESULTS_DIR, args.save_path)
    model_path = join(logs_path, "model.pth")
    device = DEVICE

    # number of copies in contrastive
    if args.contrastive:
        n_copies = args.contrastive_n_copies
        lambda_c = args.contrastive_lambda
    else:
        n_copies = 1
        lambda_c = None

    real_batch_size = args.real_bs
    equiv_batch_size = args.equiv_bs
    equiv_batch_size /= n_copies

    assert equiv_batch_size%real_batch_size == 0
    accumulation_steps = equiv_batch_size//real_batch_size
    num_classes = len(admitted_classes)
    if args.pretrained.startswith("custom"):
        preload_epoch = args.pretrained.split("_")[1]
        custom_preload_path = join(CUSTOM_PRELOAD_PATH, args.arquitecture, "checkpoints", preload_epoch, "model.pth")
    else:
        custom_preload_path=None

    training_config = {"logs_path": logs_path,
                       "model_path":model_path,
                       "epochs": args.epochs,
                       "device": device,
                       "batch_size": real_batch_size,
                       "num_classes": num_classes,
                       "accumulation_steps": accumulation_steps,
                       "custom_preload_path": custom_preload_path,
                       "contrastive": contrastive,
                       "lambda_c": lambda_c,
                       "drop_epochs": args.drop_epochs,
                       "warmup_iters": args.warmup_iter,
                       "patience": 50#100 # todo change
                       }


    # model arguments:
    num_classes = len(admitted_classes)
    n_input_channels = 1
    n_input_channels = 1

    model_config = {"name": args.architecture,
                    "pretrained": True if args.pretrained == "default" else False,
                    "num_classes": num_classes,
                    "n_input_channels": n_input_channels,
                    "group_equiv": args.group_equiv_type,
                    "z2_transition": args.n_group_blocks}

    optimizer_config =  {"optim": args.optimizer,
                         "lr":args.learning_rate,}
    optimizer_config.update(args.optimizer_kw_params)
    
    # transforms arguments:
    hflip = True if augmentation in ["base", "all", "flips", "improv"] else False
    vflip = True if augmentation in ["flips"] else False  # if rotation is admitted vflip is redundant
    rotation = 180 if augmentation in ["base", "all", "rotation", "improv"] else 0
    translation = 0.05  if augmentation in ["base", "all", "translation"] else 0.0
                        # the initial image patch is collected with size=224*2
                        # translation parameter relates to this dimension
    # TODO: UNDEFINED
    contrast = 0.5 if augmentation in ["all", "intensity"] else 0
    brightness = 0.5 if augmentation in ["all", "intensity"] else 0
    scale = 0.25 if augmentation in ["all", "scale", "improv"] else 0
    elastic_deform = True if augmentation in ["all", "elastic", "improv"] else False

    mean=0.4579
    std = .2306
    patch_size=224

    train_transform_args =  {"hflip": hflip,
                             "vflip": vflip,
                             "rotation": rotation,
                             "translation": translation,
                             "contrast": contrast,
                             "brightness": brightness,
                             "scale": scale,
                             "elastic": elastic_deform,
                             "mean": mean,
                             "std": std,
                             "size": patch_size}

    test_transform_args =  {"hflip": False,
                            "vflip": False,
                            "rotation": 0,
                            "translation": 0,
                            "contrast": 0,
                            "brightness": 0,
                            "scale": 0,
                            "elastic": 0,
                            "mean": mean,
                            "std": std,
                            "size": patch_size}

    if dataset == "CBIS":
        directory = CBIS_DATASET_DIR
    else:
        raise(ValueError("Unknown dataset argument"))

    train_dataset_config = {"directory": join(directory, "train"),
                            "admitted_classes": admitted_classes,
                            "half_size":patch_size,
                            "n_copies": n_copies,
                            "join_by_malignancy": join_by_malignancy}

    val_dataset_config = {"directory": join(directory, "val"),
                          "admitted_classes": admitted_classes,
                          "half_size":patch_size,
                          "join_by_malignancy": join_by_malignancy}

    config = {"training":            training_config,
              "model":               model_config,
              "optimizer":           optimizer_config,
              "train_get_transform": train_transform_args,
              "test_get_transform":  test_transform_args,
              "train_dataset":       train_dataset_config,
              "val_dataset":         val_dataset_config}
    return config
