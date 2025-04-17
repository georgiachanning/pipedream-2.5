import torch.nn as nn
from .model_stages import GPTBlockWrapper, GPTFinal
from .gpt_config import get_gpt_config

def stage3():
    config = get_gpt_config()
    module = nn.Sequential(
        *[GPTBlockWrapper(config) for _ in range(3)],
        GPTFinal(config)
    )
    input_names = ["hidden9"]
    output_names = ["model_pred"]
    return module, input_names, output_names
