
import torch
from torch.utils.benchmark import Timer, Compare
torch.manual_seed(0)
results = []

def test(_sizes):
    x = torch.randn(*_sizes, dtype=torch.float)
    xcpu = x.cpu()
    xcuda = x.cuda()
    tmp=[]
    
    def _subtest(stmt, desc, xcpu=xcpu, xcuda=xcuda):
        t1 = Timer(
            stmt=stmt,
            label='svd',
            sub_label=str(x.size()),
            description=desc,
            globals=dict(globals(), **locals())
            )
        tmp.append(t1.blocked_autorange().mean)
    
    _subtest('torch.linalg.svd(xcpu,  full_matrices=False)', 'cpu')
    _subtest("torch.linalg.svd(xcuda, full_matrices=False)", 'cuda')
    results.append(tmp[0]-tmp[1])

# test((100, 10, 10))
# test((300, 900, 100))
test((1000, 60, 3))
test((8716, 3, 2))
# Compare(results).print()
print(results)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
