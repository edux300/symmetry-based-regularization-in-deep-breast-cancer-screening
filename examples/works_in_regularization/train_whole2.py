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
    Xs = torch.cat(Xs)
    y = y.repeat(4).to(device)

    with torch.cuda.amp.autocast(enabled=True):
        out, repre = model(Xs, ret_repre=True)
        loss = (loss_fn(out, y.float())).mean()/ accumulation_steps
        zs_norm = F.normalize(repre, dim=1).detach()
        zm = zs_norm.view([4, -1, repre.shape[-1]]).mean(0, keepdim=True)
        #loss += contrastive_loss(repre, zm.repeat(4, 1, 1).reshape(-1, zm.shape[-1])).mean(0) * lambda_c / accumulation_steps

        # new loss: todo
        loss += contrastive_loss(predictor_contrastive(repre), zm.repeat(4, 1, 1).reshape(-1, zm.shape[-1])).mean(0) * lambda_c / accumulation_steps
        # loss += contrastive_loss(zs_norm, zm.repeat(4, 1, 1).reshape(-1, zm.shape[-1])).mean(0) * lambda_c / accumulation_steps
    resources["loss_classifier"] = loss.detach().cpu().numpy() * accumulation_steps

    scaler.scale(loss).backward()

    if (current_step+1) % accumulation_steps == 0:
        #optimizer.step()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    current_step+=1

    resources["out"] = torch.sigmoid(out).to(torch.float32)
    resources["labels"] = y

def train_fn(resources):
    global current_step
    model = resources["model"]
    optimizer = resources["optimizer"]
    device = resources["device"]
    X, y = resources["batch"]

    #y_items = [i.item() for i in y]
    #cw = torch.tensor([class_weights[i] for i in y_items]).to(device)

    model.train()

    X = X.to(device)
    y = y.to(device)

    with torch.cuda.amp.autocast(enabled=True):
        out = model(X)
        #loss = loss_fn(out, y.float()) / accumulation_steps
        
        #loss = (loss_fn(out, y.float()) * cw).mean()/ accumulation_steps
        loss = (loss_fn(out, y.float())).mean()/ accumulation_steps
        #loss = (loss_fn(out, y.float()) * class_weights[y.int()]).mean() / accumulation_steps


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

    resources["out"] = torch.sigmoid(out).to(torch.float32)
    resources["labels"] = y

def tune_bn():
    global valid_first_flag
    model.train()
    for X, y in tune_bn_loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                model(X)
    valid_first_flag=False

valid_first_flag=False
def valid_fn(resources):
    global valid_first_flag
    if valid_first_flag:
        tune_bn()
    model = resources["model"]
    device = resources["device"]
    X, y = resources["batch"]
    
    #y_items = [i.item() for i in y]
    #cw = torch.tensor([class_weights[i] for i in y_items]).to(device)

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        #loss = (loss_fn(out, y.float()) * cw).mean()
        loss = (loss_fn(out, y.float())).mean()

    resources["out"] = torch.sigmoid(out)
    resources["labels"] = y
    resources["loss_classifier"] = torch.mean(loss)

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


from torchvision.models.resnet import Bottleneck
from torch import nn
from models.momentum_batchnorm import change_model_mbn, change_momentum_only, change_model_gn
from models.conv_ws import change_model_conv2d, Conv2d


class attentionModel(nn.Module):
    def __init__(self, in_channels, w, h, embedd=512, n_keys=8):
        super().__init__()
        self.n_keys = n_keys
        self.in_channels = in_channels
        self.w = w
        self.h = h
        self.embedd = embedd
        self.scale = torch.sqrt(torch.tensor(embedd))
        self.positional_encoding = positional_encoding(w, h, embedd//2)
        self.query_linear = nn.Conv2d(in_channels, embedd, 1)
        self.keys = nn.Embedding(n_keys, embedd)
        self.first_proj = nn.Conv2d(in_channels, embedd, 1)
        self.classifier = nn.Sequential(nn.Linear(embedd*self.n_keys, embedd),
                                        nn.Dropout(.5),
                                        nn.ReLU(),
                                        nn.Linear(embedd, 1),)

    def forward(self, X):
        #self.positional_encoding = self.positional_encoding.to(X.device)
        queries = (self.query_linear(X) + self.positional_encoding.to(X.device)) / self.scale
        X = self.first_proj(X)
        #queries = torch.cat((X, self.positional_encoding), 1)
        indices = torch.arange(self.n_keys).to(X.device).view(1, -1).repeat(X.shape[0], 1)
        att = torch.einsum("nchw,nkc->nkhw", queries, self.keys(indices))
        att = torch.softmax(att.reshape(-1, self.w * self.h), 1).reshape(-1, self.n_keys, self.w, self.h)
        out = torch.einsum("nkhw,nchw->nkc", att, X).reshape([-1, self.n_keys*self.embedd])
        return out.view(X.shape[0], -1), self.classifier(out)

def positional_encoding(w=25, h=25, channels=16):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
    pos_x = torch.arange(w).type(inv_freq.type())
    pos_y = torch.arange(h).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
    emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
    emb = torch.zeros((w, h, channels * 2))
    emb[:, :, : channels] = emb_x
    emb[:, :, channels : 2 * channels] = emb_y
    return emb.view([-1, *emb.shape]).permute([0, 3, 1, 2])

class RegionGroupPooling(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()
        self.layer = nn.Linear(in_channels, 1)
        self.k = k
        self.in_channels = in_channels

    def forward(self, X):
        bs = X.shape[0]
        X = X.permute([0, 2, 3, 1])
        out = self.layer(X)
        out = out.view(bs, -1)
        _, idxs = torch.topk(out, self.k)
        idxs = idxs.view(bs, self.k, 1).repeat(1, 1, self.in_channels)
        out = torch.gather(X.view(bs, -1, self.in_channels), 1, idxs)
        feat = torch.mean(out, 1)
        return feat, self.layer(feat)


class avg(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        self.pre = nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                 torch.nn.Flatten(),)
        self.model = torch.nn.Linear(backbone_channels, 1)

    def forward(self, X):
        repre = self.pre(X)
        out = self.model(repre)
        return repre, out

class maxp(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        self.pre = nn.Sequential(torch.nn.AdaptiveMaxPool2d((5, 5)),
                                 torch.nn.Flatten(),
                                 torch.nn.Linear(25*backbone_channels, 512),)
                                 
        self.model = nn.Sequential(torch.nn.ReLU(),
                                   torch.nn.Dropout(0.5),
                                   torch.nn.Linear(512, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(0.5),
                                   torch.nn.Linear(512, 1))

    def forward(self, X):
        repre = self.pre(X)
        out = self.model(repre)
        return repre, out
        

class WholeImageModel(nn.Module):
    def __init__(self, backbone, backbone_channels, k, pooling_type="avg"):
        super().__init__()
        self.backbone = backbone.features

        if pooling_type=="avg":
            self.rpg = avg(backbone_channels)
        if pooling_type=="max5":
            self.rpg = maxp(backbone_channels)
        elif pooling_type=="rgp":
            self.rpg = RegionGroupPooling(backbone_channels, k)
        """
        elif pooling_type=="att"
            self.rpg = attentionModel(backbone_channels, 25, 25)
        """

    def forward(self, x, ret_repre=False):
        out = self.backbone(x)
        out = nn.functional.relu(out, inplace=True)
        repre, out = self.rpg(out)
        out = out.view(-1)

        if ret_repre:
            return out, repre
        return out

if __name__ == "__main__":
    import os
    import yaml
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import torch.nn as nn

    #from models.classification import get_model
    from models.classification.init_equivariant import get_model
    from optimizers.optimizers import get_optimizer
    #from data_loaders.mammography.whole_image_dataset import WholeImageDatasetCBIS, WholeImageDatasetCMMD
    from data_loaders.mammography.whole_image_dataset import WholeImageDatasetCBISSimple, WholeImageDatasetCMMD
    from data_loaders.data_augmentation import simple_large_whole_image_tranform

    from engine.main import MainCallback, train_epoch
    from engine.warmup_lr import WarmupLR
    from engine.logger import Logger, FileEpochLogger
    from engine.metrics import AverageCallback, AccuracyCallback, BalancedAccuracyCallback, RocAUCCallback, F1ScoreCallback
    from engine.save import saveMultiBestKModel, saveLastModelCallback
    from engine.monitoring import WeightNormMonitoring

    from examples.works_in_regularization.config_whole2 import get_config

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
    
    backbone = get_model(**config["model"])

    
    if config["model"]["z2_transition"]>0 and config["training"]["preload_path"] == "":
        state_dict = torch.load("../External_drive/basemodels/hybrid.pth")
        state_dict.pop("classifier.weight")
        state_dict.pop("classifier.bias")
        #print(state_dict["features.conv0.weight"])
        with torch.no_grad():
            weight = torch.sum(state_dict["features.conv0.weight"], dim=1, keepdim=True)
            state_dict["features.conv0.weight"] = weight

        #print(state_dict["features.conv0.weight"])
        backbone.load_state_dict(state_dict, strict=False)
    
    with torch.no_grad():
        _, in_channels, H, W = backbone.features(torch.rand(2, 1, *config["test_get_transform"]["size"])).shape
    k = int(H*W*0.5)
    

    model = WholeImageModel(backbone, in_channels, k=k, pooling_type=config["training"]["pooling_type"])
    model = model.to(device)
    print(model)

    if config["training"]["preload_path"] != "":
        state_dict = torch.load(config["training"]["preload_path"])
        print(f"Preloading model: {config['training']['preload_path']}")
        model.load_state_dict(state_dict)

    # transforms
    train_transform = simple_large_whole_image_tranform(**config["train_get_transform"])
    val_transform = simple_large_whole_image_tranform(**config["test_get_transform"])
    print(train_transform)
    print(val_transform)

    # datasets
    if config["training"]["dataset"] == "CBIS":
        train_config = config["dataset"].copy()
        val_config = config["dataset"].copy()
        train_config["directory"] += "/train"
        val_config["directory"] += "/val"
        
        train_dataset = WholeImageDatasetCBISSimple(**train_config, transform=train_transform, n_copies=config["training"]["n_copies"])
        tune_dataset = WholeImageDatasetCBISSimple(**train_config, transform=val_transform)
        val_dataset = WholeImageDatasetCBISSimple(**val_config, transform=val_transform)
    elif config["training"]["dataset"] == "INbreast":  
        train_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=train_transform, n_copies=config["training"]["n_copies"])
        train_dataset.fold_split(config["training"]["fold"], "train")
        tune_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=val_transform)
        tune_dataset.fold_split(config["training"]["fold"], "train")
        val_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=val_transform)
        val_dataset.fold_split(config["training"]["fold"], "test")
    elif config["training"]["dataset"] == "CMMD":
        train_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=train_transform, n_copies=config["training"]["n_copies"])
        train_dataset.fold_split(config["training"]["fold"], "train_train")
        tune_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=val_transform)
        tune_dataset.fold_split(config["training"]["fold"], "train_train")
        val_dataset = WholeImageDatasetCBISSimple(**config["dataset"], transform=val_transform)
        val_dataset.fold_split(config["training"]["fold"], "train_val")
    print("Datasets Length")
    print(len(train_dataset))
    print(len(tune_dataset))
    print(len(val_dataset))

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["training"]["batch_size"],
                                               drop_last=True,
                                               num_workers=2,
                                               shuffle=True,
                                               pin_memory=True)
    
    tune_bn_loader = torch.utils.data.DataLoader(tune_dataset,
                                                 batch_size=config["training"]["batch_size"],
                                                 shuffle=True,
                                                 drop_last=True,
                                                 num_workers=2,
                                                 pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config["training"]["batch_size"],
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # optimizer
    """
    optimizer = get_optimizer(**config["optimizer"])([{"params":model.rpg.parameters(),
                                                       "lr":config["optimizer"]["lr"],
                                                       "weight_decay":1e-5},
                                                      {"params":model.backbone.parameters(),
                                                       "lr":config["optimizer"]["lr"]/5,
                                                       "weight_decay":1e-5}])
    """
    #if not config["model"]["z2_transition"]>0:
    optimizer = get_optimizer(**config["optimizer"])(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:min((x+1)/512, 1))
    #else:
    #optimizer = get_optimizer(optim="SGD",
    #                          lr=0.01,
    #                          momentum=0.9,
    #                          weight_decay=1e-5)(model.parameters())
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:10**(-(x//(0.25*30*len(train_loader)))))

    #class_weights = train_dataset.class_weight
    print(optimizer)

    if config["training"]["contrastive"]:
        contrastive_loss = lambda z1, z2 : -F.cosine_similarity(z1, z2)
        lambda_c = config["training"]["lambda_c"]
        """
        #dim = 1664
        dim = 512 if config["training"]["pooling_type"]=="max5" else 1664
        pred_dim = 512
        predictor_contrastive = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                              nn.BatchNorm1d(pred_dim),
                                              nn.ReLU(inplace=True), # hidden layer
                                              nn.Linear(pred_dim, dim)) # output layer
        predictor_contrastive.to(device)
        optimizer.add_param_group({"params": predictor_contrastive.parameters(),
                                   "lr":0.01})        
        """
        predictor_contrastive = lambda x:x

    accumulation_steps = config["training"]["accumulation_steps"]
    scaler = torch.cuda.amp.GradScaler()
    current_step = 0

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 45, gamma=0.25)
    #scheduler = None
    
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:1)

    resources = dict()
    resources["model"]  = model
    resources["optimizer"] = optimizer
    resources["device"] = device
    resources["train_data_loader"] = train_loader
    resources["val_data_loader"] = val_loader

    losses = ["loss_classifier"]
    metrics = ["accuracy", "bal-accuracy", "rocAUC", "f1-score"]

    callbacks = [AverageCallback(loss, phase="both") for loss in losses]
    
    metrics_callbacks = [AccuracyCallback(phase="both", softmax_first = False),
                         BalancedAccuracyCallback(phase="both", softmax_first = False),
                         RocAUCCallback(phase="both", softmax_first = False),
                         F1ScoreCallback(phase="both", softmax_first = False)]

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

    sel_tr_func = train_fn if not config["training"]["contrastive"] else train_fn_contrastive
    #patience = 15 if not config["training"]["contrastive"] else 4
    #counter=0
    max_patience = 40
    if config["training"]["contrastive"]:
        max_patience *= 0.25
    patience = max_patience
    max_rocAUC = 0
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.25, patience=patience,cooldown=1)
    
    for i in range(epochs):
        valid_first_flag = True
        train_epoch(i, sel_tr_func, valid_fn, resources, callbacks)
        plot_lines()
        if resources["average_valid_rocAUC"] > max_rocAUC:
            max_rocAUC = resources["average_valid_rocAUC"]
            patience = max_patience
        else:
            patience -= 1
        if patience==0 and config["training"]["enable_stopping"]:
            break

        """
        if scheduler is not None:
            scheduler.step(resources["average_valid_rocAUC"])
            if scheduler.in_cooldown:
                counter+=1
            if counter==4:
                break
        """
        """
        if i==init_train_epoch:
            optimizer.add_param_group({"params":model.backbone.parameters(),
                                       "lr":config["optimizer"]["lr"]/5,
                                       "weight_decay":1e-5})
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode="max",
                                                                   factor=0.25,
                                                                   patience=patience,
                                                                   cooldown=1)
        """

    print("Finished")
