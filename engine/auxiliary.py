import yaml
import torch
from os.path import dirname, abspath, join
import sys

def load_config(config_name=None, terminal=True, safe=True):
    with open(config_name) as file:
        if safe:
            config = yaml.safe_load(file)
        else:
            config = yaml.load(file)
    if terminal:
        config.update(load_terminal_args())
    return config

def load_terminal_args():
    kw_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            kw_dict[key] = value
    return kw_dict
        
def to_numpy(a):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return a