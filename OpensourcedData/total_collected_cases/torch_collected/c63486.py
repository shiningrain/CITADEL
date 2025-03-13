
import torch

torch.distributed.init_process_group("mpi")

device = torch.device("cuda")  # bug does not appear on cpu

rank = torch.distributed.get_rank()
assert torch.distributed.get_world_size() == 2
other_worker = (rank + 1) % 2

for _ in range(10):
    handles = []

    # send two messages to the other worker
    for message in [1, 2]:  # there seems to be no bug if I send only one message
        payload = torch.tensor([10 * rank + message], device=device)  # the first digit indicates the sender
        handles.append(torch.distributed.isend(payload, dst=other_worker, tag=message))

    # receive the messages
    results = []
    for message in [1, 2]:
        recv_buffer = torch.tensor([-1], device=device)  # buffer is initialized at -1
        handles.append(torch.distributed.irecv(recv_buffer, src=other_worker, tag=message))
        results.append(recv_buffer)

    # wait for send and receive operations to finish
    for handle in handles:
        handle.wait()

    # print the received values
    if rank == 0:
        print([r.item() for r in results])  # expecting [11, 12]
