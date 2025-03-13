
# import matplotlib.pyplot as plt
import torch

model = torch.nn.Linear(2,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min='1e-6')

lrs = []
lrs_opt = []
for _ in range(100):
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    lrs_sched.append(scheduler.get_last_lr())
    lrs_opt.append(optimizer.param_groups[0]["lr"])

# plt.plot(lrs_sched, color='green',)
# plt.show()
# plt.plot(lrs_opt, color='red')
# plt.show()
 