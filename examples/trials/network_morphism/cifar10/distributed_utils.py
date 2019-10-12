import os
import torch
import torch.distributed as dist
from torch.nn import Module
import multiprocessing as mp
from nni.networkmorphism_tuner.graph import TorchModel

class DistModule(TorchModel):
    def __init__(self, module):
        super(DistModule, self).__init__(module.graph)
        self.module = module
        broadcast_params(self.module)
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    addr = node_list
    print("Addr = " + str(addr))
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr #Only use one node
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
