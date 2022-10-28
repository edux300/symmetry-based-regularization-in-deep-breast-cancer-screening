import random
import numpy as np

"""
"optimizer": {"optim": optim,
              "lr":learning_rate,
              "weight_decay": weight_decay,
              "eps": eps,
              "betas": betas},
"""

def get_config():
    # training
    logs_path = f"/media/emcastro/External_drive/results/pretrain_imagenet/densenet169/"
    model_path = logs_path+"/model.pth"
    data_path = "../datasets/imagenet/"
    epochs = 90
    num_workers = 4
    device = "cuda:0"
    batch_size = 32
    accumulation_steps = 8
    drop_iterations = [1.5e5, 3e5]

    # optimizer
    learning_rate = .1
    optim = "SGD"
    weight_decay = 1e-4
    momentum = .9

    # model
    model_name = "densenet169"

    num_classes = 1000
    n_input_channels = 3
    resume = True

    config = {"training": {"logs_path": logs_path,
                           "model_path":model_path,
                           "data_path":data_path,
                           "epochs": epochs,
                           "num_workers":num_workers,
                           "device": device,
                           "batch_size": batch_size,
                           "num_classes": num_classes,
                           "accumulation_steps": accumulation_steps,
                           "drop_iterations": drop_iterations,
                           "resume": resume
                           },


              "optimizer": {"optim": optim,
                            "lr":learning_rate,
                            "weight_decay": weight_decay,
                            "momentum": momentum
                            },
              
              "model": {"name": model_name,
                        "pretrained": True,
                        "num_classes": num_classes,
                        "n_input_channels": n_input_channels,
                        "group_equiv": "p4",
                        "z2_transition": 4}
             }

    return config
