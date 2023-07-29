import torch
import json

     
def allot_device(random_seed_value):
    if torch.cuda.is_available():
        device = "cuda"
        torch.manual_seed(random_seed_value)
    else:
        device = "cpu"
    return device


# load config data
def load_config():
    with open("Configs.json") as json_file:
        data = json.load(json_file)
    return data 

