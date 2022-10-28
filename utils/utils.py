"""
author: Eduardo Castro

Set of auxiliary functions

Functions:
    
    load_config - load a configuration from a yaml file
        args:
            config_file - config file path
            terminal - whether to include terminal arguments in the config
    
    load_terminal_args - load terminal key word arguments onto a dictionary
"""

import yaml
import sys

def load_config(config_file, terminal=True):        
    with open(config_file) as file:
        config = yaml.safe_load(file)
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
        