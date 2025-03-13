
import torch
import random
import numpy as np

def set_seed(seed): 
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

"""
Test consistency of network optimization when using PyTorch's CUDA Graph API
"""
N, D_in, H, D_out = 640, 4096, 2048, 1024

# define 'conventional' network
set_seed(1)
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                        torch.nn.ReLU(),
                        torch.nn.Linear(H, D_out),
                        torch.nn.ReLU()).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# define network to be accelerated
set_seed(1)
model_cg = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                        torch.nn.ReLU(),
                        torch.nn.Linear(H, D_out),
                        torch.nn.ReLU()).cuda()
loss_fn_cg = torch.nn.MSELoss()
optimizer_cg = torch.optim.Adam(model_cg.parameters(), lr=0.1)
    
# check consistent weight initialization
assert (torch.allclose(model_cg[0].weight.detach().clone(), model[0].weight.detach().clone())), "Initialized Network weights in layer 0 are not the same!\n" \
                                                                                                f"Weights using CUDAGraph:\n {model_cg[0].weight.detach().clone()}\n" \
                                                                                                f"Weights not using CUDAGraph:\n {model[0].weight.detach().clone()}"
assert (torch.allclose(model_cg[0].bias.detach().clone(), model[0].bias.detach().clone())), "Initialized Network bias in layer 0 are not the same!\n" \
                                                                                                f"Bias using CUDAGraph:\n {model_cg[0].bias.detach().clone()}\n" \
                                                                                                f"Bias not using CUDAGraph:\n {model[0].bias.detach().clone()}"
assert (torch.allclose(model_cg[2].weight.detach().clone(), model[2].weight.detach().clone())), "Initialized Network weights in layer 2 are not the same!\n" \
                                                                                                f"Weights using CUDAGraph:\n {model_cg[2].weight.detach().clone()}\n" \
                                                                                                f"Weights not using CUDAGraph:\n {model[2].weight.detach().clone()}"
assert (torch.allclose(model_cg[2].bias.detach().clone(), model[2].bias.detach().clone())), "Initialized Network bias in layer 2 are not the same!\n" \
                                                                                                f"Bias using CUDAGraph:\n {model_cg[2].bias.detach().clone()}\n" \
                                                                                                f"Bias not using CUDAGraph:\n {model[2].bias.detach().clone()}"

# Placeholders used for capture
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# run optimizer.step() once, such that it is consistent with warmup run
optimizer.zero_grad(set_to_none=True)
y_pred = model(static_input)
loss = loss_fn(y_pred, static_target)
loss.backward()
optimizer.step()

# warmup - note that optimizer_cg.step() is called once
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(1):
        optimizer_cg.zero_grad(set_to_none=True)
        y_pred_cg = model_cg(static_input)
        loss_cg = loss_fn_cg(y_pred_cg, static_target)
        loss_cg.backward()
        optimizer_cg.step()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
optimizer_cg.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred_cg = model_cg(static_input)
    static_loss_cg = loss_fn_cg(static_y_pred_cg, static_target)
    static_loss_cg.backward()
    optimizer_cg.step()

# check consistency
assert (torch.allclose(model_cg[0].weight.detach().clone(), model[0].weight.detach().clone())), "Network weights in layer 0 after warmup are not the same!\n" \
                                                                                                f"Weights using CUDAGraph:\n {model_cg[0].weight.detach().clone()}\n" \
                                                                                                f"Weights not using CUDAGraph:\n {model[0].weight.detach().clone()}"
assert (torch.allclose(model_cg[0].bias.detach().clone(), model[0].bias.detach().clone())), "Network bias in layer 0 after warmup are not the same!\n" \
                                                                                                f"Bias using CUDAGraph:\n {model_cg[0].bias.detach().clone()}\n" \
                                                                                                f"Bias not using CUDAGraph:\n {model[0].bias.detach().clone()}"
assert (torch.allclose(model_cg[2].weight.detach().clone(), model[2].weight.detach().clone())), "Network weights in layer 2 after warmup are not the same!\n" \
                                                                                                f"Weights using CUDAGraph:\n {model_cg[2].weight.detach().clone()}\n" \
                                                                                                f"Weights not using CUDAGraph:\n {model[2].weight.detach().clone()}"
assert (torch.allclose(model_cg[2].bias.detach().clone(), model[2].bias.detach().clone())), "Network bias in layer 2 after warmup are not the same!\n" \
                                                                                                f"Bias using CUDAGraph:\n {model_cg[2].bias.detach().clone()}\n" \
                                                                                                f"Bias not using CUDAGraph:\n {model[2].bias.detach().clone()}"

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # compare 'conventional' forward, backward and step 
    optimizer.zero_grad(set_to_none=True)
    y_pred = model(data)
    loss = loss_fn(y_pred, target)
    loss.backward()
    optimizer.step()

    # Fill the graph's input memory with new data to compute on
    static_input.copy_(data)
    static_target.copy_(target)
    # replay() includes forward, backward, and step.
    g.replay()

    # check consistency
    assert (torch.allclose(model_cg[0].weight.detach().clone(), model[0].weight.detach().clone())), "Network weights in layer 0 are not the same!\n" \
                                                                                                    f"Weights using CUDAGraph:\n {model_cg[0].weight.detach().clone()}\n" \
                                                                                                    f"Weights not using CUDAGraph:\n {model[0].weight.detach().clone()}"
    assert (torch.allclose(model_cg[0].bias.detach().clone(), model[0].bias.detach().clone())), "Network bias in layer 0 are not the same!\n" \
                                                                                                    f"Bias using CUDAGraph:\n {model_cg[0].bias.detach().clone()}\n" \
                                                                                                    f"Bias not using CUDAGraph:\n {model[0].bias.detach().clone()}"
    assert (torch.allclose(model_cg[2].weight.detach().clone(), model[2].weight.detach().clone())), "Network weights in layer 2 are not the same!\n" \
                                                                                                    f"Weights using CUDAGraph:\n {model_cg[2].weight.detach().clone()}\n" \
                                                                                                    f"Weights not using CUDAGraph:\n {model[2].weight.detach().clone()}"
    assert (torch.allclose(model_cg[2].bias.detach().clone(), model[2].bias.detach().clone())), "Network bias in layer 2 are not the same!\n" \
                                                                                                    f"Bias using CUDAGraph:\n {model_cg[2].bias.detach().clone()}\n" \
                                                                                                    f"Bias not using CUDAGraph:\n {model[2].bias.detach().clone()}"
