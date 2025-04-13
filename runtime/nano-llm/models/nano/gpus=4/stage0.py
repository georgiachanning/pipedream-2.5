import torch.nn as nn
from .model_stages import GPTEmbedding, GPTBlockWrapper  # adjust relative import if needed
from .config import get_gpt_config

def stage0():
    config = get_gpt_config()
    module = nn.Sequential(
        GPTEmbedding(config),
        *[GPTBlockWrapper(config) for _ in range(3)]
    )
    input_names = ["input0"]
    output_names = ["hidden3"]
    return module, input_names, output_names
