
import torch
d = "cuda:0"
#d = "cpu"
x = [pow(10,x) for x in list(range(10))]
for size in x:
    data = torch.rand(size,1)
    lin = torch.nn.Linear(1,1,bias=False).to(d)
    #lin.weight.data = torch.tensor([[1.0]])
    print (size,torch.norm(data.to(d)*lin.weight.to(d) - lin.to(d)(data.to(d))).item())
