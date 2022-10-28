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
RESULTS_DIR = os.environ["DEEPMM_RESULTSDIR"] + "/Works_on_Regularization_new/"
#CBIS_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/cbis_whole_image_clean/"
#CBIS_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/test_join/"
CBIS_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/cbis_whole_image_no_repeat_adjust/"
CMMD_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/CMMD_classification/"
INbreast_DATASET_DIR = os.environ["DEEPMM_DATADIR"] + "/inbreast_classification/"
DEVICE = "cuda:0"

"""
python3 examples/works_in_regularization/train.py --arquitecture p4_64 --pretrained custom_20 --optimization scratch_AdamW --data CBIS_mass --augmentation base;
"""

def argparse_config():
    argparser = argparse.ArgumentParser(description='Deep Learning Regularization Parameters')
    argparser.add_argument('--architecture', type=str, default="densenet169", help='options: vgg16, resnet34, resnet50, densenet121, convnext_tiny, efficientnet_b4')
    argparser.add_argument('--group_equiv_type', type=str, default="p4")
    argparser.add_argument('--n_group_blocks', type=int, default=0, help='number_of_equivariant_blocks')

    #argparser.add_argument('--optimizer', type=str, default="SGD", help='SGD, Adam, AdamW')
    #argparser.add_argument('--learning_rate', type=float, default=5e-03, help='effective batch size (accumulation)')
    #argparser.add_argument('--optimizer_kw_params', type=json.loads, default='{"nesterov":true, "momentum":0.9, "weight_decay":0.00005}')
    argparser.add_argument('--pooling_type', type=str, default='avg')
    argparser.add_argument('--optimizer', type=str, default="Adam", help='SGD, Adam, AdamW')
    argparser.add_argument('--learning_rate', type=float, default=2e-05, help='effective batch size (accumulation)')
    argparser.add_argument('--optimizer_kw_params', type=json.loads, default='{"weight_decay":0.00005}')

    argparser.add_argument('--epochs', type=int, default=250, help='number of epochs')
    argparser.add_argument('--real_bs', type=int, default=16, help='batch size')
    argparser.add_argument('--equiv_bs', type=int, default=32, help='effective batch size (accumulation)')

    argparser.add_argument('--data', type=str, default='CBIS', help='supported: CBIS')
    argparser.add_argument('--contrastive', default=False, action='store_true')
    argparser.add_argument('--contrastive_lambda', type=float, default=1.0)
    argparser.add_argument('--contrastive_n_copies', type=int, default=4)
    argparser.add_argument('--augmentation', default='base', type=str, help='supported: base, elastic')
    argparser.add_argument('--save_path', default='whole_image_temp', type=str, help='allows multiple runs')
    
    argparser.add_argument('--fold', type=int, default=1, help='supported: 1,2,3,4,5')
    argparser.add_argument('--preload_path', type=str, default='', help='Paths to the pretrained model')
    return argparser.parse_args()

def get_config(parse=True):
    args = argparse_config()

    dataset = args.data
    contrastive = args.contrastive
    augmentation = args.augmentation

    assert args.optimizer in ["SGD", "Adam", "AdamW"]

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
    num_classes = 2

    training_config = {"logs_path": logs_path,
                       "model_path":model_path,
                       "epochs": args.epochs,
                       "device": device,
                       "batch_size": real_batch_size,
                       "num_classes": num_classes,
                       "accumulation_steps": accumulation_steps,
                       "contrastive": contrastive,
                       "lambda_c": lambda_c,
                       "dataset": dataset,
                       "n_copies": n_copies,
                       "pooling_type": args.pooling_type,
                       "preload_path": args.preload_path,
                       "fold": args.fold,
                       "enable_stopping": dataset != "INbreast"
                       }

    # model arguments:
    n_input_channels = 1

    model_config = {"name": args.architecture,
                    "pretrained": True,
                    "num_classes": num_classes,
                    "n_input_channels": n_input_channels,
                    "group_equiv": args.group_equiv_type,
                    "z2_transition": args.n_group_blocks}

    optimizer_config =  {"optim": args.optimizer,
                         "lr":args.learning_rate,}
    optimizer_config.update(args.optimizer_kw_params)

    mean = 0.27
    std = 0.24
    size = (800, 800)

    # transforms arguments:
    """
    hflip = True
    vflip = False
    rotation = 30
    translation = 0.05
    brightness = 0.25
    scale = 0.25
    contrast = 0.25
    assert augmentation in ["base", "elastic"]
    elastic_deform = False if augmentation == "base" else True
    """
    hflip = True
    vflip = False
    rotation = 25
    translation = 0.05
    brightness = 0.0
    scale = 0.2
    contrast = 0.0
    assert augmentation in ["base", "elastic"]
    elastic_deform = False if augmentation == "base" else True

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
                             "size": size}

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
                            "size": size}

    if dataset == "CBIS":
        directory = CBIS_DATASET_DIR
    elif dataset == "CMMD":
        directory = CMMD_DATASET_DIR
    elif dataset == "INbreast":
        directory = INbreast_DATASET_DIR
    else:
        raise(ValueError("Unknown dataset argument"))
    
    
    dataset_config = {"directory": directory}
    

    config = {"training":            training_config,
              "model":               model_config,
              "optimizer":           optimizer_config,
              "train_get_transform": train_transform_args,
              "test_get_transform":  test_transform_args,
              "dataset":             dataset_config}
    return config
