import yaml
import os

def load_config(config_file=None):
    if config_file is None:
        raise ValueError("No configuration file has been specified")
        #config_file=os.path("../config.yaml")
    with open(config_file, "r") as conf:
        config =  yaml.load(conf,Loader=yaml.SafeLoader)
    return config

def validate_config():
    #TODO
    return

