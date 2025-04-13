import torch.nn as nn
from .model_stages import GPTBlockWrapper
from .config import get_gpt_config

def stage2():
    config = get_gpt_config()
    module = nn.Sequential(
        *[GPTBlockWrapper(config) for _ in range(3)]
    )
    input_names = ["hidden6"]
    output_names = ["hidden9"]
    return module, input_names, output_names
