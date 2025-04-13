from nanoGPT.model import GPTConfig

def get_gpt_config():
    return GPTConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )
