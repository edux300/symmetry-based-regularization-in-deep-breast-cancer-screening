"""
Mammography Patch Classification
"""
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import os
import pickle as pkl

# Creates once at the beginning of training
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

def save_checkpoint(epoch, config, model, optimizer, iteration):
    path = config["training"]["logs_path"]
    os.makedirs(os.path.join(path, "checkpoints", str(epoch)))
    model_path = os.path.join(path, "checkpoints", str(epoch), "model.pth")
    optim_path = os.path.join(path, "checkpoints", str(epoch), "optim.pth")
    state_path = os.path.join(path, "checkpoints", str(epoch), "state.pkl")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    state = {"iterations": iteration, "epoch": epoch}
    with open(state_path, "wb") as file:
        pkl.dump(state, file)

def load_checkpoint(epoch, config, model, optimizer):
    path = config["training"]["logs_path"]
    model_path = os.path.join(path, "checkpoints", str(epoch), "model.pth")
    optim_path = os.path.join(path, "checkpoints", str(epoch), "optim.pth")
    state_path = os.path.join(path, "checkpoints", str(epoch), "state.pkl")
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optim_path))
    with open(state_path, "rb") as file:
        state = pkl.load(file)
    return state


if __name__ == "__main__":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import yaml
    import sys
    import torch.nn as nn
    from torch.optim.lr_scheduler import MultiStepLR
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    from models.classification.init_equivariant import get_model
    from optimizers.optimizers import get_optimizer

    from engine.main import MainCallback, train_epoch
    from engine.logger import Logger, FileEpochLogger
    from engine.metrics import AverageCallback
    from engine.metrics import BacthAccuracyCallback, BacthTop5AccuracyCallback

    from examples.works_in_regularization.pretrain_config import get_config

    # configuration file
    config = get_config()

    # training variables
    logs_path = config["training"]["logs_path"]
    epochs = config["training"]["epochs"]
    device = torch.device(config["training"]["device"])

    if not config["training"]["resume"]:
        os.makedirs(logs_path)
        with open(logs_path+"config.yaml", 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    # model
    model = get_model(**config["model"])
    model = model.to(device)
    print(model)

    # Data loading code
    traindir = os.path.join(config["training"]["data_path"], 'train')
    valdir = os.path.join(config["training"]["data_path"], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            normalize,
        ]))


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["training"]["batch_size"],
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=config["training"]["num_workers"],
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # optimizer
    optimizer = get_optimizer(**config["optimizer"])(model.parameters())
    print(optimizer)

    initial_epoch = 0
    initial_iteration = 0
    if config["training"]["resume"]:
        from glob import glob
        from os.path import basename
        epoch = basename(sorted(glob(os.path.join(config["training"]["logs_path"], "checkpoints", "*")), key=lambda x:int(x.split("/")[-1]))[-1])
        state = load_checkpoint(epoch, config, model, optimizer)
        initial_epoch = state["epoch"]
        #initial_iteration = state["iterations"]
        # mistake between step and iteration
        initial_iteration = 5004*initial_epoch


    accumulation_steps = config["training"]["accumulation_steps"]
    scaler = torch.cuda.amp.GradScaler()
    current_step = initial_iteration * accumulation_steps

    scheduler = MultiStepLR(optimizer, milestones=list(config["training"]["drop_iterations"]), gamma=0.1)
    for _ in range(initial_iteration):
        scheduler.step()

    resources = dict()
    resources["model"]  = model
    resources["optimizer"] = optimizer
    resources["device"] = device
    resources["train_data_loader"] = train_loader
    resources["val_data_loader"] = val_loader

    losses = ["loss_classifier"]
    metrics = ["accuracy", "top5-accuracy"]

    callbacks = [AverageCallback(loss, phase="both") for loss in losses]
    callbacks.append(BacthAccuracyCallback(phase="both"))
    callbacks.append(BacthTop5AccuracyCallback(phase="both"))

    keys = []
    for k in losses+metrics:
        keys.append("average_train_{}".format(k))
        keys.append("average_valid_{}".format(k))

    callbacks.append(Logger(keys))
    callbacks.append(FileEpochLogger(config["training"]["logs_path"], keys))
    callbacks = MainCallback(callbacks)

    for i in range(initial_epoch+1, epochs):
        train_epoch(i, train_fn, valid_fn, resources, callbacks)
        if i % 5 == 0:
            save_checkpoint(i, config, model, optimizer, current_step//accumulation_steps)

    print("Finished")
