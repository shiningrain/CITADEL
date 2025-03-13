
# import matplotlib.pyplot as plt
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

model = torch.nn.Linear(2,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = func_cls(optimizer, T_0=10, T_mult=2, eta_min='1e-6')

for _ in range(100):
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

# plt.plot(lrs_sched, color='green',)
# plt.show()
# plt.plot(lrs_opt, color='red')
# plt.show()
 