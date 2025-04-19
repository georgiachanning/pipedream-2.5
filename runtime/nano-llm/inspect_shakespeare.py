#!/usr/bin/env python3
import os
import numpy as np
from shakespeare_wrapper import ShakespeareDataset

data_dir   = "/homes/cdt24/cgeorgia/projects/opt/pipe-nanoGPT/data/shakespeare_char"
train_bin  = os.path.join(data_dir, "train.bin")

print("1) Quick memmap check of raw train.bin with different dtypes:")
for dt in (np.uint8, np.int16, np.int32, np.int64):
    try:
        arr = np.memmap(train_bin, dtype=dt, mode='r')
        mx  = int(arr.max())
        print(f"   dtype={dt.__name__:7s} → max token ID = {mx:5d}")
    except Exception as e:
        print(f"   dtype={dt.__name__:7s} → error {e!r}")
print()

print("2) Now scan via your ShakespeareDataset wrapper:")
block_size = 1024
ds = ShakespeareDataset(data_dir, block_size=block_size, split='train')
print("   len(ds) =", len(ds), "(# of [batch, block_size] chunks)")

max_id = -1
for xb, yb in ds:
    mx = int(xb.max())
    if mx > max_id:
        max_id = mx
    # also check targets just in case...
    mx2 = int(yb.max())
    if mx2 > max_id:
        max_id = mx2

print(f"   max token ID returned by Dataset = {max_id}")
print(f"   => you need vocab_size ≥ {max_id+1}")
