
import torch
import torch.fft as fft
import numpy as np
from scipy.fft import rfft
np.random.seed(42)
torch.manual_seed(0)

if __name__ == '__main__':
    dtype = torch.float64
    for device in ['cuda']:
        print('\ndevice={}'.format(device))
        Ly = 1600
        y = torch.randn(1, 1, Ly, dtype=dtype, device=device)
        for Lx in [16001]:
            x = torch.randn(1, 1, Lx, dtype=dtype, device=device)
            x = torch.nn.functional.pad(x, [0, Ly-1])
            yy = torch.nn.functional.pad(y, [0, Lx-1])
            X = fft.rfft(x)
            Y = fft.rfft(yy)
            X_s = rfft(x[0, 0, :].cpu().numpy())
            Y_s = rfft(yy[0, 0, :].cpu().numpy())
            dif_x = np.max(np.abs(X_s - X[0, 0, :].cpu().numpy()))
            dif_y = np.max(np.abs(Y_s - Y[0, 0, :].cpu().numpy()))
            # print('Lx={}: dif_x={}, dif_y={}'.format(Lx, dif_x, dif_y))
            print(dif_x)
    
    save_path='./tmp_result.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(r_e_s, f)
