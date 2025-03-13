import torch
import torch.nn.functional as F
from torch.testing import make_tensor

torch.manual_seed(0)

torch.backends.cudnn.allow_tf32 = True

x = make_tensor(1, 2 * 4, 5, 5, dtype=torch.float32, device='cuda')
w = make_tensor(2 * 4, 8, 3, 3, dtype=torch.float32, device='cuda')

result = F.conv_transpose2d(x, w, groups=2).double()
expected = F.conv_transpose2d(x.double(), w.double(), groups=2)

amax = result.sub(expected).div(expected).abs().argmax()
print("% difference: ", result.sub(expected).div(expected).abs().max())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)