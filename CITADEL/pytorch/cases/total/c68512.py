
import torch
import time

torch.manual_seed(0)

###
nb_trials = 10
kernel_size = 3
dilation = 2**8
nb_chs = 64
batch_size = 16
data_length = 28

devices = ['cuda:0']

dims = {
    '2d':{
        'tensor' : torch.rand(batch_size,nb_chs,data_length,data_length),
        'conv' : torch.nn.Conv2d,
    },
}

def get_time_perfs(nb_trials, verbose=True):
    result=[]
    for device in devices:
        for dim in dims.keys():

            a = dims[dim]['tensor']
            a = a.to(device)

            
            for pad in ['zeros', 'replicate']:

                conv = dims[dim]['conv'](nb_chs,nb_chs, kernel_size=kernel_size, padding_mode=pad,
                    padding=dilation*(kernel_size-1)//2, dilation=dilation)
                conv.to(device)

                if 'cuda' in device:
                    torch.cuda.synchronize(device=device)
                time_start = time.time()
                for _ in range(nb_trials):
                    b = conv(a)
                    b = b.detach().cpu()
                if 'cuda' in device:
                    torch.cuda.synchronize(device=device)
                time_end = time.time()

                result.append(1000*(time_end-time_start)/nb_trials)
                # if verbose:
                #     print(f"{device}, {dim}, {pad} took {1000*(time_end-time_start)/nb_trials} ms")

    return result[1]/result[0]

get_time_perfs(2, verbose=False)
print(get_time_perfs(nb_trials))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
