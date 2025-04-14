from nanoGPT.model import GPTConfig

def get_gpt_config():
    ## bigger GPT

    return GPTConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    ## baby GPT
    ## for shakespeare char

    '''return GPTConfig(
        vocab_size=65,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        bias=False
    )'''
    
