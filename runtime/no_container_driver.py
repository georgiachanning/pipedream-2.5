#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import subprocess
import datetime
import uuid

# required keys in your config.yml
LOG_DIR           = 'log_directory'
MODULE            = 'module'
DATA_DIR          = 'data_dir'
MACHINES          = 'machines'
MODEL_TYPE        = 'model_type'
DISTRIBUTED_BACKEND = 'distributed_backend'
CONFIG_FILE       = 'config_file'
GROUP_ID = uuid.uuid4().hex


class WorkerInfo:
    def __init__(self, ip, gpu_id):
        self.ip = ip
        self.gpu_id = int(gpu_id)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', required=True, help='path to your config.yml')
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    args = p.parse_args()

    # load config
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    # sanity checks
    for k in (LOG_DIR, MODULE, DATA_DIR, MACHINES, MODEL_TYPE, DISTRIBUTED_BACKEND):
        assert k in cfg, f"Missing `{k}` in config.yml"

    # build workers list
    workers = []
    for m in cfg[MACHINES]:
        ip, gpu = m.split(':')
        workers.append(WorkerInfo(ip, gpu))

    # make a timestamped log directory
    output_dir = os.path.join(cfg[LOG_DIR], datetime.datetime.now().isoformat())
    os.makedirs(output_dir, exist_ok=True)

    # copy config into logs
    subprocess.check_output(f"cp {args.config_file} {output_dir}", shell=True)

    num_ranks = len(workers)
    for rank, w in enumerate(workers):
        # build the command
        cmd = [
            'python', 'nano-llm/main_with_runtime.py',
            f'--data_dir={cfg[DATA_DIR]}',
            f'--module={cfg[MODULE]}',
            f'--distributed_backend={cfg[DISTRIBUTED_BACKEND]}',
            f'--config_path={cfg.get(CONFIG_FILE, "")}',
            f'--rank={rank}',
            f'--local_rank={rank % num_ranks}',
            f'-b {cfg.get("batch_size", 16)}'
        ]
        if args.resume:
            cmd.append(f'--resume={args.resume}')

        cmd_str = " ".join(cmd) + f" 2>&1 | tee {output_dir}/worker{rank}.log"

        # if the machine isn't localhost, ssh to it
        if w.ip not in ('localhost', '127.0.0.1'):
            launch = f'ssh -n {w.ip} "{cmd_str}"'
        else:
            launch = cmd_str

        print(f"[rank {rank:>2}] -> {launch}")
        # fire-and-forget each worker
        subprocess.Popen(launch, shell=True)
