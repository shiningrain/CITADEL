
import torch
input_tensor = torch.randint(-1,1,[3], dtype=torch.int64)
input = input_tensor.clone()
r = 100
print(torch.combinations(input, r=r))
# killed
