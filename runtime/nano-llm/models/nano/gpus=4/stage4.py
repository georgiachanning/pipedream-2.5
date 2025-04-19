from .model_stages import GPTLossWrapper
import torch.nn as nn

def stage4(criterion):
    loss_module = GPTLossWrapper(criterion)
    input_names = ["model_pred"]
    output_names = ["loss"]
    return loss_module, input_names, output_names

