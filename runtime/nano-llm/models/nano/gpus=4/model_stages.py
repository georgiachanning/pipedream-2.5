import torch
import torch.nn as nn
import sys
sys.path.append("/homes/cdt24/cgeorgia/projects/opt/")

from nanoGPT.model import GPT, Block, LayerNorm

class GPTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        idx = idx.long()
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)

class GPTBlockWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = Block(config)

    def forward(self, x):
        return self.block(x)

class GPTFinal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        x = self.ln_f(x)
        return self.lm_head(x)
        
class GPTLossWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, target):
        logits = logits.view(-1, logits.size(-1))    # [batch*seq, vocab]
        target = target.view(-1)                     # [batch*seq]
        return self.criterion(logits, target)
