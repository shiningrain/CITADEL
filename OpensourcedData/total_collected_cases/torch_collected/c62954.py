
import torch
import torch.nn as nn
import math

class FFNExpert(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation_fn = nn.functional.relu):
        super().__init__()
        torch.set_printoptions(precision=16)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)

    def forward(self, x: torch.tensor):
        x = self.linear1(x)
        return x

class MergedFFNExpert(nn.Module):
    def __init__(self, d_model=8, dim_feedforward=64, local_num_experts=2, activation_fn = nn.functional.relu):
        super().__init__()
        torch.set_printoptions(precision=16)
        self.weight1 = nn.Parameter(torch.Tensor(local_num_experts, d_model, dim_feedforward))
        with torch.no_grad():
            # make initialization the same with FFNExpert
            for i in range(local_num_experts):
                wshape = self.weight1[i].shape
                nn.init.kaiming_uniform_(self.weight1[i].view(wshape[1], wshape[0]), a=math.sqrt(5))
                self.weight1[i] = self.weight1[i].view(wshape[1], wshape[0]).t().detach().clone()
        self.activation_fn = activation_fn
        self.local_num_experts = local_num_experts
        print(f"MergedFFNExpert self.weight1.shape {self.weight1.shape}")

    def forward(self, x: torch.tensor):
        input_shape = x.shape
        reshaped_x = x.reshape(input_shape[1], input_shape[0], input_shape[2], input_shape[3])
        reshaped_x = x.reshape(reshaped_x.shape[0], -1, reshaped_x.shape[-1])
        out1 = torch.bmm(reshaped_x, self.weight1)
        print(f"out1.shape {out1.shape}")
        return out1

torch.set_printoptions(precision=18)
device = 'cuda'
# device = 'cpu'
print(f"Device type: {device}")
d_model=8
dim_feedforward=64
n_local_expert = 2

# 0. init MergedFFNExpert
torch.manual_seed(1033)
merged_ffn_expert = MergedFFNExpert(d_model=d_model, dim_feedforward=dim_feedforward, local_num_experts=n_local_expert).to(device)

# 0. init FFNExpert
torch.manual_seed(1033)
ffn_experts = nn.ModuleList()
for i in range(n_local_expert):
    ffn_experts.append(FFNExpert(d_model, dim_feedforward).to(device))

# # 1.check if weight of ffn_expert and merged_ffn_expert can pass allclose (need to reshape the weights)
# ffn_expert_weights_linear1 = []
# ffn_expert_weights_linear2 = []
# with torch.no_grad():
    # for i in range(n_local_expert):
        # ffn_expert_weights_linear1 += [ffn_experts[i].linear1.weight.transpose(1, 0)]
# ffn_expert_linear1_all = torch.cat(ffn_expert_weights_linear1, dim=0)
# wshape = merged_ffn_expert.weight1.shape
# ffn_expert_linear1_all = ffn_expert_linear1_all.reshape(wshape[0], wshape[1], wshape[2])
# merged_ffn_expert_weight1 = merged_ffn_expert.weight1
# assert torch.allclose(ffn_expert_linear1_all, torch.where(merged_ffn_expert_weight1 > 0, merged_ffn_expert_weight1, ffn_expert_linear1_all)),"faied to assert allclose of  ffn_expert_linear1_all and merged_ffn_expert_weight1 result  "
# print(f"ffn_expert_linear1_all and merged_ffn_expert_weight1 result allclose: {torch.allclose(ffn_expert_linear1_all, torch.where(merged_ffn_expert_weight1 > 0, merged_ffn_expert_weight1, ffn_expert_linear1_all))}")

# 2.a prepare the input
dispatched_input = torch.randn(1, 2, 1248, 8).to(device)
print(f"input.shape {dispatched_input.shape}")

# 2.b get the forward output of ffn_experts
merged_ffn_expert_output = merged_ffn_expert(dispatched_input)
torch.set_printoptions(precision=16)


# 2.c get the forward output of ffn_experts
torch.manual_seed(1033)
ffn_experts = nn.ModuleList()
for i in range(n_local_expert):
    ffn_experts.append(FFNExpert(d_model, dim_feedforward).to(device))
chunks = dispatched_input.chunk(n_local_expert, dim=1)
ffn_expert_outputs = []

for chunk, expert in zip(chunks, ffn_experts):
    ffn_expert_outputs += [expert(chunk)]
ffn_expert_output = torch.cat(ffn_expert_outputs, dim=1)



# 3 check allclose the forward output of ffn_experts
print(f"FFN and MergedFFN result allclose: {torch.allclose(merged_ffn_expert_output, torch.where(ffn_expert_output > 0, ffn_expert_output, merged_ffn_expert_output))}")
