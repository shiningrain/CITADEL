
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
d = "cuda:0"
#d = "cpu"
x=7
x = [pow(10,x)]# for x in list(range(10))
for size in x:
    data = torch.rand(size,1)
    lin = func_cls(out_features=1, bias=False).to(d)
    #lin.weight.data = torch.tensor([[1.0]])
    r_e_s=torch.norm(data.to(d)*lin.weight.to(d) - lin.to(d)(data.to(d))).item()
    save_path='./tmp_result.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(r_e_s, f)