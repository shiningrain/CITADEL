.encodings attribute is lost.
The following is a minimize implementation:
 python
import argparse
from torch import nn
import torch
from icecream import ic
import os
import torch.distributed as dist

from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__

        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print
def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser



class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('ckpt/roberta-base')
        self.linear = nn.Linear(1,10)
    def forward(self,text=None,tokenized=None,xe=None):
        if tokenized:
            #tokenized._encodings = xe
            data = self.linear(tokenized.input_ids.float().unsqueeze(-1))
            return tokenized, data
        tokenized = self.tokenizer(text, padding="longest", return_tensors="pt").to('cuda')
        data = self.linear(tokenized.input_ids.float().unsqueeze(-1))
        return tokenized,data
        
if __name__=='__main__':
    args = get_args_parser().parse_args()

    init_distributed_mode(args)
    model = M()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    tokenizer = RobertaTokenizerFast.from_pretrained('ckpt/roberta-base')
    
    text = ['I dont know','I know']

    tokenized = tokenizer(text, padding="longest", return_tensors="pt").to('cuda')
    ic(tokenized._encodings)
    ic(id(tokenized))
    tokenized,_ = model(tokenized = tokenized)
    ic(tokenized._encodings)
    ic(id(tokenized))
