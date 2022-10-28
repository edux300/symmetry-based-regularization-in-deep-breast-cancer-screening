import torch

def get_optimizer(optim, **kwargs):
    if optim == "SGD":
        return lambda x: torch.optim.SGD(x, **kwargs)
    if optim == "Adam":
        return lambda x: torch.optim.Adam(x, **kwargs)
    if optim == "AdamW":
        return lambda x: torch.optim.AdamW(x, **kwargs)