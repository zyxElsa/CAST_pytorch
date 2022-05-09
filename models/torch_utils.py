import os
import random

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def concat_all_gather(tensor, world_size):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_rank(group=None):
    try:
        return torch.distributed.get_rank(group)
    except:
        return 0


def get_world_size(group=None):
    try:
        return torch.distributed.get_world_size(group)
    except:
        return 1


def kaiming_init(mod):
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
        if mod.weight.requires_grad:
            nn.init.kaiming_normal_(mod.weight, a=0.2, mode="fan_in")
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def update_average(net, net_ema, m=0.999):
    net = net.module if hasattr(net, "module") else net
    for p, p_ema in zip(net.parameters(), net_ema.parameters()):
        p_ema.data.mul_(m).add_((1.0 - m) * p.detach().data)


def warmup_learning_rate(optimizer, lr, train_step, warmup_step):
    if train_step > warmup_step or warmup_step == 0:
        return lr
    ratio = min(1.0, train_step/warmup_step)
    lr_w = ratio * lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_w
    return lr_w
