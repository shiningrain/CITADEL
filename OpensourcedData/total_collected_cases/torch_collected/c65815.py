
import torch

def custom_norm(tensor):
    return torch.sqrt((tensor**2).sum())

def test_norm(N):
    A = torch.ones((N,N))
    print(f"Size: {A.shape} \t:: Custom norm: {custom_norm(A)} \t:: Torch norm {torch.linalg.norm(A)}")

test_norm(10)
test_norm(100)
test_norm(1000)
test_norm(10000)
 
Output is 

Size: torch.Size([10, 10]) 	:: Custom norm: 10.0 	:: Torch norm 10.0
Size: torch.Size([100, 100]) 	:: Custom norm: 100.0 	:: Torch norm 100.0
Size: torch.Size([1000, 1000]) 	:: Custom norm: 1000.0 	:: Torch norm 1000.0
Size: torch.Size([10000, 10000]) 	:: Custom norm: 10000.0 	:: Torch norm 4096.0


## Expected behavior
The torch norm is wrong in the last line, probably due to some internal overflow (?). 
Expected behavior would be to report that the capacity of the datatype is not high enough, e.g. raise some Error.

Then the user can correct it to the following, which will yield the correct output.


import torch
def custom_norm(tensor):
    return torch.sqrt((tensor**2).sum())
def test_norm(N):
    A = torch.ones((N,N),dtype=torch.float64)
    print(f"Size: {A.shape} \t:: Custom norm: {custom_norm(A)} \t:: Torch norm {torch.linalg.norm(A)}")
test_norm(10)
test_norm(100)
test_norm(1000)
test_norm(10000)
