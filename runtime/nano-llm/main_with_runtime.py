# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time
import wandb
import math

import torch
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.utils import clip_grad_norm_

sys.path.append("..")
sys.path.append("/homes/cdt24/cgeorgia/projects/opt/pipe-nanoGPT")
import adam
import adamw
import nadam
import runtime
import sgd
from shakespeare_wrapper import ShakespeareDataset

parser = argparse.ArgumentParser(description='Pipedream nano-GPT Training')
parser.add_argument('--data_dir', type=str, 
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--grad_clip', '--gc', default=0.0, type=float,
                    help='grad_clip default 0.0')
parser.add_argument('--warmup_iters', type=int, default=100,
                    help='number of steps for linear warmup')
parser.add_argument('--min_lr', type=float, default=None,
                    help='minimum LR after cosine decay (default: base_lr/10)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')
# Adding functionality to let you choose optimizer
parser.add_argument('--optimizer_key', default='default', choices=['adam', 'adamw','nadam', 'default'],
                    help='Choices are "adam", "nadam", or "default", which will give you vanilla SGD.')
# Adding w&b tracking
parser.add_argument('--wandb', action='store_true', help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='opt')
parser.add_argument('--wandb_entity', type=str, default='georgia-channing-university-of-oxford')

best_prec1 = 0


# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    # input and target are both token IDs with shape (batch_size, block_size)
    block_size = 256 # or whatever you use
    input_size = [args.batch_size, block_size]

    training_tensor_shapes = {
        "input0": input_size,
        "target": input_size
    }

    # training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]} 
    dtypes = {"input0": torch.int64, 
              "target": torch.int64}

    inputs_module_destinations = {"input0": 0, "target":4} # changed to input0 from input and added target
    target_tensor_names = {"target"}
    for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
        input_tensors = []
        for input in inputs:
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=torch.float32)
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [args.eval_batch_size] + training_tensor_shapes[key][1:])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.IMAGE_CLASSIFICATION,
        enable_recompute=args.recompute)

    all_modules = r.modules()
    print(f"PipeDream loaded {len(all_modules)} modules:")
    for i, m in enumerate(all_modules):
        print(f"  [{i:>2}] {m.__class__.__name__}")

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks

    # commented this out to see if preventing learning
    '''if not is_first_stage():
        args.synthetic_data = True'''
    
    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    if args.wandb and is_last_stage():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            id=os.environ.get('WANDB_RUN_ID'),       # unique per worker
            group=os.environ.get('WANDB_RUN_GROUP'), # same for all workers
            save_code=True
        )

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    
    if args.optimizer_key == 'adam':
        optimizer = adam.AdamWithWeightStashing(
                        modules=r.modules(), master_parameters=r.master_parameters,
                        model_parameters=r.model_parameters, loss_scale=args.loss_scale,
                        num_versions=num_versions, 
                        lr=args.lr, 
                        betas=(0.9,0.99),
                        weight_decay=args.weight_decay, 
                        verbose_freq=args.verbose_frequency,
                        macrobatch=args.macrobatch)

    if args.optimizer_key == 'adamw':
        optimizer = adamw.AdamWWithWeightStashing(
                        modules=r.modules(), master_parameters=r.master_parameters,
                        model_parameters=r.model_parameters, loss_scale=args.loss_scale,
                        num_versions=num_versions, 
                        lr=args.lr, 
                        betas=(0.9,0.99),
                        weight_decay=args.weight_decay, 
                        verbose_freq=args.verbose_frequency,
                        macrobatch=args.macrobatch)
    elif args.optimizer_key == 'nadam':
        optimizer = nadam.NAdamWithWeightStashing(
                        modules=r.modules(), master_parameters=r.master_parameters,
                        model_parameters=r.model_parameters, loss_scale=args.loss_scale,
                        num_versions=num_versions, 
                        lr=args.lr, 
                        betas=(0.9,0.999),
                        weight_decay=args.weight_decay, 
                        verbose_freq=args.verbose_frequency,
                        macrobatch=args.macrobatch,
                        eps=1e-08, momentum_decay=0.004)
    else:
        optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                        r.model_parameters, args.loss_scale,
                        num_versions=num_versions,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        verbose_freq=args.verbose_frequency,
                        macrobatch=args.macrobatch)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_path = os.path.join(args.data_dir)  # should contain train.bin, val.bin

    train_dataset = ShakespeareDataset(data_path, block_size=block_size, split='train')
    val_dataset = ShakespeareDataset(data_path, block_size=block_size, split='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train(train_loader, r, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, r, epoch)
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, args.checkpoint_dir, r.stage)


def train(train_loader, r, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.train(n)
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    num_warmup_minibatches = args.warmup_iters

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()

    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        r.run_forward()

        # Adjust learning 
        adjust_learning_rate(optimizer, epoch, args.epochs, r, args.lr_policy, i, n)

        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss = r.output, r.target, r.loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            top1.update(prec1[0], output.size(0))
            top5.update(prec5[0], output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i+1)) * float(n)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, n, batch_time=batch_time,
                       epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                       loss=losses, top1=top1, top5=top5,
                       memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_reserved()) / 10**9)))

                if args.wandb:
                    wandb.log({
                        'val/loss': losses.val,
                        'val/prec@1': top1.val,
                        'val/prec@5': top5.val,
                        'val/batch_time':batch_time.val,
                        'avg/loss': losses.avg,
                        'avg/prec@1': top1.avg,
                        'avg/prec@5': top5.avg,
                        'avg/batch_time':batch_time.avg,
                        'epoch': epoch,
                        'memory': float(torch.cuda.memory_allocated()) / 10**9,
                        'epoch_time':epoch_time
                    })

                import sys; sys.stdout.flush()
        else:
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                       epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_reserved()) / 10**9)))
                import sys; sys.stdout.flush()


        # perform backward pass
        if args.fp16:
            r.zero_grad()
        else:
            optimizer.zero_grad()
        optimizer.load_old_params()
        r.run_backward()
        optimizer.load_new_params()

        if args.grad_clip != 0.0:
            clip_grad_norm_(r.model_parameters, args.grad_clip)

        optimizer.step()

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        optimizer.zero_grad()
        optimizer.load_old_params()
        r.run_backward()
        optimizer.load_new_params()

        if args.grad_clip != 0.0:
            clip_grad_norm_(r.model_parameters, args.grad_clip)
            
        optimizer.step()

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1[0], output.size(0))
                top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_reserved()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


def save_checkpoint(state, checkpoint_dir, stage):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar" % stage)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, total_epochs, r,
                         lr_policy, step, epoch_length):
    """
    Adjusts learning rate based on stage, epoch, and policy.

    Supported LR policies:
         - step
         - polynomial
         - exponential_decay
         - cosine
    """

    # 1) base LR (already stage‑adjusted)
    base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

    # 2) global step / total steps
    current_step = epoch * epoch_length + step
    max_steps    = total_epochs * epoch_length

    # 3) warm‑up
    warmup_steps = getattr(args, "warmup_iters", 0)
    if warmup_steps > 0 and current_step < warmup_steps:
        lr = base_lr * float(current_step + 1) / float(warmup_steps)
    else:
        # 4) post‑warmup decay
        if lr_policy == "step":
            lr = base_lr * (0.1 ** (epoch // 30))

        elif lr_policy == "polynomial":
            power = 2.0
            lr = base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)

        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))

        elif lr_policy == "cosine":
            # determine min_lr (if user didn’t set it, default to base_lr/10)
            if getattr(args, "min_lr", None) is None:
                min_lr = base_lr * 0.1
            else:
                min_lr = args.min_lr

            denom = max_steps - warmup_steps
            # make sure we don’t go negative if warmup_steps > current_step
            t = max(0, current_step - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * t / denom))
            lr = min_lr + (base_lr - min_lr) * cosine_decay

        else:
            raise NotImplementedError(f"Unknown lr_policy: {lr_policy}")

    # 5) log and set
    if step % 100 == 0:
        print(f"Epoch: {epoch} Step {step}\tLearning rate: {lr:.6g}")

    for pg in optimizer.param_groups:
        pg['lr'] = lr

'''def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy, step, epoch_length):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

    if args.lr_warmup and epoch < 5:
        lr = stage_base_lr * float(1 + step + epoch*epoch_length)/(5.*epoch_length)
    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.1 ** (epoch // 30))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
        else:
            raise NotImplementedError

    if step % 100 == 0:
        print("Epoch: %d Step %d \tLearning rate: %f" % (epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr'''

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for language modeling by flattening batch and seq dimensions."""
    with torch.no_grad():
        # output: [B, T, vocab_size]
        B, T, V = output.size()
        # flatten batch and sequence dimensions
        output = output.reshape(B * T, V)  # [B*T, vocab_size]
        target = target.reshape(B * T)     # [B*T]

        maxk = max(topk)
        # Get the topk predictions for each element in the flattened output
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # pred: [B*T, maxk]
        pred = pred.t()  # Now shape is [maxk, B*T]
        # Expand target to match the shape of pred for comparison
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Compute the number of correct predictions for top k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / (B * T)))
        return res


if __name__ == '__main__':
    main()
