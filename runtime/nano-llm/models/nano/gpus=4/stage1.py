import torch.nn as nn
from .model_stages import GPTBlockWrapper
from .gpt_config import get_gpt_config

def stage1():
    config = get_gpt_config()
    module = nn.Sequential(
        *[GPTBlockWrapper(config) for _ in range(3)]
    )
    input_names = ["hidden3"]
    output_names = ["hidden6"]
    return module, input_names, output_names
