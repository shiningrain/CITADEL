
import torch
torch.manual_seed(0)
import torch.multiprocessing as mp
import torch.distributed as dist

result=[]

def main_worker(rank, world_size, args):
    global result
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:

        q_buffer = torch.rand([4,4]).to(0)      
        Q,R = torch.linalg.qr(q_buffer)
        # print("local before q_buffer",Q)
        result.append(Q.cpu().detach().numpy())
        dist.send(Q,1)


                    


    elif rank == 1:

        q_buffer = torch.rand([4,4]).to(1)

        dist.recv(q_buffer,0)

        # print("recv",q_buffer)
        result.append(q_buffer.cpu().detach().numpy())


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
    print(result[0]-result[1])

    # save_path='./tmp_result.pkl'
    # with open(save_path, 'wb') as f:
    #     pickle.dump(r_e_s, f)