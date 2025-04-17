#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import subprocess
import datetime
import uuid

# required keys in your config.yml
LOG_DIR             = 'log_directory'
MODULE              = 'module'
DATA_DIR            = 'data_dir'
MACHINES            = 'machines'
MODEL_TYPE          = 'model_type'
DISTRIBUTED_BACKEND = 'distributed_backend'
CONFIG_FILE         = 'config_file'

# generate one group ID for all W&B runs
GROUP_ID = uuid.uuid4().hex

class WorkerInfo:
    def __init__(self, ip, gpu_id):
        self.ip = ip
        self.gpu_id = int(gpu_id)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True,
                   help='path to your driver config.yml')
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    args = p.parse_args()

    # load the YAML
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    # sanity check
    for k in (LOG_DIR, MODULE, DATA_DIR, MACHINES,
              MODEL_TYPE, DISTRIBUTED_BACKEND, CONFIG_FILE):
        assert k in cfg, f"Missing `{k}` in config.yml"

    # build worker list
    workers = []
    for m in cfg[MACHINES]:
        ip, gpu = m.split(':')
        workers.append(WorkerInfo(ip, gpu))

    # make a timestamped log directory
    output_dir = os.path.join(cfg[LOG_DIR],
                              datetime.datetime.now().isoformat())
    os.makedirs(output_dir, exist_ok=True)

    # copy this YAML into logs for posterity
    subprocess.check_output(f"cp {args.config_file} {output_dir}", shell=True)

    # --- build the “base” command from required fields ---
    base_cmd = [
        'python', 'nano-llm/main_with_runtime.py',
        f'--data_dir={cfg[DATA_DIR]}',
        f'--module={cfg[MODULE]}',
        f'--distributed_backend={cfg[DISTRIBUTED_BACKEND]}',
        f'--config_path={cfg[CONFIG_FILE]}',
    ]

    # optional flags: YAML key -> CLI flag
    optional_flags = {
        'batch_size':           '-b',
        'learning_rate':        '--lr',
        'learning_rate_policy': '--lr_policy',
        'weight_decay':         '--weight-decay',
        'epochs':               '--epochs',
        'print_frequency':      '--print-freq',
        'no_input_pipelining':  '--no_input_pipelining',
        'verbose_frequency':    '--verbose',
        'lr_warmup':            '--lr_warmup',
        'synthetic_data':       '--synthetic_data',
        'recompute':            '--recompute',
        'macrobatch':           '--macrobatch',
    }

    for key, flag in optional_flags.items():
        if key in cfg:
            val = cfg[key]
            if isinstance(val, bool):
                if val:
                    base_cmd.append(flag)
            else:
                base_cmd.append(f"{flag} {val}")

    # W&B settings from YAML
    if 'wandb_project' in cfg:
        base_cmd.append('--wandb')
        base_cmd.append(f"--wandb_project={cfg['wandb_project']}")
    if 'wandb_entity' in cfg:
        base_cmd.append(f"--wandb_entity={cfg['wandb_entity']}")

    # resume
    if args.resume:
        base_cmd.append(f"--resume={args.resume}")

    num_ranks = len(workers)

    # launch one worker per GPU
    for rank, w in enumerate(workers):
        # give each its own run ID
        os.environ['WANDB_RUN_GROUP'] = GROUP_ID
        os.environ['WANDB_RUN_ID']    = f"{GROUP_ID}-{rank}"

        cmd = base_cmd.copy()
        cmd.append(f"--rank={rank}")
        cmd.append(f"--local_rank={rank % num_ranks}")

        cmd_str = " ".join(cmd) + f" 2>&1 | tee {output_dir}/worker{rank}.log"

        if w.ip not in ('localhost', '127.0.0.1'):
            launch = f'ssh -n {w.ip} "{cmd_str}"'
        else:
            launch = cmd_str

        print(f"[rank {rank:>2}] -> {launch}")
        subprocess.Popen(launch, shell=True)
