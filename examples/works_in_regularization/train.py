"""
Mammography Patch Classification
"""
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
# Creates once at the beginning of training



def train_fn_contrastive(resources):
    global current_step
    model = resources["model"]
    optimizer = resources["optimizer"]
    device = resources["device"]
    Xs, y = resources["batch"]

    model.train()

    Xs = [X.to(device) for X in Xs]
    y = y.to(device)

    with torch.cuda.amp.autocast():
        zs = []
        loss = 0
        for X in Xs:
            out, z = model.forward_with_representations(X)
            loss += loss_fn(out, y)
            zs.append(z)
        zs_norm = [F.normalize(z, dim=1) for z in zs]
        zm = torch.stack(zs_norm, 0).mean(0).detach()

        for z in zs:
            loss += contrastive_loss(z, zm).mean(0) * lambda_c

        loss /= len(Xs)
        loss /= accumulation_steps
        """
        outa, za = model.forward_with_representations(Xa)
        outb, zb = model.forward_with_representations(Xb)

        loss = loss_fn(outa, y)
        loss_addon = contrastive_loss(za, zb) * lambda_c
        loss+= loss_addon
        loss/= accumulation_steps
        """

    resources["loss_classifier"] = loss.detach().cpu().numpy() * accumulation_steps

    scaler.scale(loss).backward()

    if (current_step+1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler.warming_up():
            scheduler.step()

    current_step+=1

    resources["out"] = out.to(torch.float32)
    resources["labels"] = y


def train_fn(resources):
    global current_step
    model = resources["model"]
    optimizer = resources["optimizer"]
    device = resources["device"]
    X, y = resources["batch"]

    model.train()

    X = X.to(device)
    y = y.to(device)

    with torch.cuda.amp.autocast():
        out = model(X)
        loss = loss_fn(out, y) / accumulation_steps


    resources["loss_classifier"] = loss.detach().cpu().numpy() * accumulation_steps

    scaler.scale(loss).backward()
    #loss.backward()

    if (current_step+1) % accumulation_steps == 0:
        #optimizer.step()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler.warming_up():
            scheduler.step()

    current_step+=1

    resources["out"] = out.to(torch.float32)
    resources["labels"] = y

def valid_fn(resources):
    model = resources["model"]
    device = resources["device"]
    X, y = resources["batch"]

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = loss_fn(out, y)

    resources["out"] = out
    resources["labels"] = y
    resources["loss_classifier"] = torch.mean(loss)

def add_group(model, optimizer, k=0):
    if k==0:
        layers = [model.fc]
    elif k==1:
        layers = [model.layer1[1], model.layer1[2], model.layer2, model.layer3, model.layer4]
    elif k==2:
        layers = [model.conv1, model.bn1, model.layer1[0]]
    params = []
    for layer in layers:
        params.extend(list(layer.parameters()))
    optimizer.add_param_group({"params": params})


def plot_lines():
    from matplotlib import pyplot as plt
    from numpy import genfromtxt
    keys = losses + metrics
    for key in keys:
        plt.figure()
        for reg in ["train", "valid"]:
            file = os.path.join(logs_path, f"average_{reg}_{key}.csv")
            array = genfromtxt(file, delimiter=",")
            plt.plot(array, label=reg)
        plt.title(key)
        plt.legend()
        plt.savefig(os.path.join(logs_path, f"plot_{key}.png"))
        plt.close()

if __name__ == "__main__":
    import os
    import yaml
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import torch.nn as nn

    #from models.classification import get_model
    from models.classification.init_equivariant import get_model, add_forward_with_representations
    from optimizers.optimizers import get_optimizer
    from data_loaders.mammography.sampling_dataset import PresampledImageDataset
    from data_loaders.data_augmentation import get_transform

    from engine.main import MainCallback, train_epoch
    from engine.warmup_lr import WarmupLR
    from engine.logger import Logger, FileEpochLogger
    from engine.metrics import AverageCallback, AccuracyCallback, BalancedAccuracyCallback, RocAUCCallback, F1ScoreCallback
    from engine.save import saveMultiBestKModel, saveLastModelCallback
    from engine.monitoring import WeightNormMonitoring

    from examples.works_in_regularization.config import get_config

    # configuration file
    config = get_config()

    # training variables
    logs_path = config["training"]["logs_path"]
    epochs = config["training"]["epochs"]
    device = torch.device(config["training"]["device"])

    os.makedirs(logs_path)
    with open(os.path.join(logs_path,"config.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # model
    model = get_model(**config["model"])

    """
    if config["training"]["custom_preload_path"] is not None:
        state_dict = torch.load(config["training"]["custom_preload_path"])
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        weight = state_dict["conv1.weight"].detach()
        weight = torch.sum(weight, dim=1).unsqueeze(1)
        state_dict["conv1.weight"].data = weight.detach()
        model.load_state_dict(state_dict, strict=False)
    """

    model = model.to(device)
    print(model)

    # transforms
    train_transform = get_transform(**config["train_get_transform"])
    val_transform = get_transform(**config["test_get_transform"])
    print(train_transform)
    print(val_transform)

    # datasets
    train_dataset = PresampledImageDataset(**config["train_dataset"], transform=train_transform)
    for i, k in enumerate(train_dataset.class_dict.keys()):
        print(f"{k} - {i}")

    val_dataset = PresampledImageDataset(**config["val_dataset"], transform=val_transform)
    n_samples = sum(train_dataset.class_weight.values())
    class_weights = [n_samples / (config["training"]["num_classes"] * x) for x in train_dataset.class_weight.values()]

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["training"]["batch_size"],
                                               drop_last=True,
                                               num_workers=2,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config["training"]["batch_size"],
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=torch.tensor(class_weights).to(device))

    # optimizer
    optimizer = get_optimizer(**config["optimizer"])(model.parameters())
    print(optimizer)


    if config["training"]["contrastive"]:
        contrastive_loss = lambda z1, z2 : -F.cosine_similarity(z1, z2)
        lambda_c = config["training"]["lambda_c"]
        model = add_forward_with_representations(model)

    accumulation_steps = config["training"]["accumulation_steps"]
    scaler = torch.cuda.amp.GradScaler()
    current_step = 0

    
    warmup_iters = config["training"]["warmup_iters"]
    #milestones = [x*(len(train_loader)//accumulation_steps)-warmup_iters for x in config["training"]["drop_epochs"]]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           patience=config["training"]["patience"],
                                                           verbose=True,
                                                           cooldown=1)
    scheduler = WarmupLR(scheduler, 0, warmup_iters)

    resources = dict()
    resources["model"]  = model
    resources["optimizer"] = optimizer
    resources["device"] = device
    resources["train_data_loader"] = train_loader
    resources["val_data_loader"] = val_loader

    losses = ["loss_classifier"]
    metrics = ["accuracy", "bal-accuracy", "rocAUC", "f1-score"]

    callbacks = [AverageCallback(loss, phase="both") for loss in losses]
    metrics_callbacks = [AccuracyCallback(phase="both"),
                         BalancedAccuracyCallback(phase="both"),
                         RocAUCCallback(phase="both"),
                         F1ScoreCallback(phase="both")]
    #callbacks.append(AccuracyCallback(phase="both"))
    #callbacks.append(BalancedAccuracyCallback(phase="both"))
    #callbacks.append(RocAUCCallback(phase="both"))
    #callbacks.append(F1ScoreCallback(phase="both"))
    callbacks.extend(metrics_callbacks)
    callbacks.append(WeightNormMonitoring(model, config["training"]["logs_path"]))

    keys = []
    for k in losses+metrics:
        keys.append("average_train_{}".format(k))
        keys.append("average_valid_{}".format(k))

    callbacks.append(Logger(keys))
    callbacks.append(FileEpochLogger(config["training"]["logs_path"], keys))
    save_model_callback = saveMultiBestKModel(config["training"]["model_path"],
                                              model, 1,
                                              [f"average_valid_{m}" for m in metrics])
    callbacks.append(save_model_callback)
    callbacks.append(saveLastModelCallback(config["training"]["model_path"], model))
    callbacks = MainCallback(callbacks)

    dropped = 0
    for i in range(epochs):
        train_epoch(i, train_fn if not config["training"]["contrastive"] else train_fn_contrastive,
                    valid_fn, resources, callbacks)
        plot_lines()
        if not scheduler.warming_up():
            scheduler.step(resources["average_valid_rocAUC"])
            if scheduler.cooldown_counter == scheduler.cooldown:
                dropped += 1
                if dropped == 3:#2:
                    # todo change
                    break

    print("Finished")
