
import torch.distributed as dist
from torch.multiprocessing import spawn


def main(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:23456", rank=rank, world_size=world_size)

    object_list = [None] * world_size

    obj = "hello" if rank == 0 else "world"
    dist.all_gather_object(object_list, obj)
    print(object_list)


if __name__ == "__main__":
    world_size = 2
    spawn(main, (world_size,), nprocs=world_size)
