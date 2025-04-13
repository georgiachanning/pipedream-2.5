
from .stage0 import stage0
from .stage1 import stage1
from .stage2 import stage2
from .stage3 import stage3
from .stage4 import stage4

def arch():
    return "nanoGPT"

def model(criterion):
    # You can ignore the passed-in criterion here if your loss stage takes care of it,
    # or alternatively you might pass it to stage4 if you want to override the default.
    stages = [stage0(), stage1(), stage2(), stage3(), stage4()]
    return stages


"""
import torch
import torch.nn as nn
import os
import sys

sys.path.append("/homes/cdt24/cgeorgia/projects/opt/")

from nanoGPT.model import GPT, GPTConfig
from .model_stages import GPTEmbedding, GPTBlockWrapper, GPTFinal, GPTLossWrapper

# Must return the model name (just a string identifier)
def arch():
    return "nanoGPT"

# Must return a list of (stage_module, input_names, output_names) tuples
def model(criterion):
    # Match your nanoGPT config
    config = GPTConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    stages = []

    # Stage 0: Embedding + Blocks 0-2
    stages.append((
        nn.Sequential(
            GPTEmbedding(config),
            *[GPTBlockWrapper(config) for _ in range(3)]
        ),
        ["input0"], ["hidden3"]
    ))

    # Stage 1: Blocks 3-5
    stages.append((
        nn.Sequential(
            *[GPTBlockWrapper(config) for _ in range(3)]
        ),
        ["hidden3"], ["hidden6"]
    ))

    # Stage 2: Blocks 6-8
    stages.append((
        nn.Sequential(
            *[GPTBlockWrapper(config) for _ in range(3)]
        ),
        ["hidden6"], ["hidden9"]
    ))

    # Stage 3: Blocks 9-11 + ln_f + lm_head
    stages.append((
        nn.Sequential(
            *[GPTBlockWrapper(config) for _ in range(3)],
            GPTFinal(config)
        ),
        ["hidden9"], ["logits"]
    ))

    # Final stage adds loss
    # stages.append((criterion, ["output", "target"], ["loss"]))

    # Stage 4
    loss_module = GPTLossWrapper(nn.CrossEntropyLoss())
    stages.append((loss_module, ["logits", "target"], ["loss"]))


    return stages
"""