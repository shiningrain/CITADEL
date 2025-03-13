
import torch
from torch.fft import fftn,fftshift

#torch.set_default_dtype(torch.double)
device="cuda"
A=torch.rand(1,1,64,32,32)
B=A.expand((1,100,64,32,32)).clone().to(device=device)/64/32/32  # B is same along the second axis

kernel1=torch.rand(64,32,32,30).to(device=device)
kernel2=torch.rand(64,32,32,30).to(device=device)
B=fftshift(B,dim=(-3,-2,-1))

C=0
for i in range(30):
    f1=B*kernel1[:,:,:,i]
    f2=B*kernel2[:,:,:,i]
    f11=fftn(f1,dim=(-3,-2,-1),norm="backward")
    f22=fftn(f2,dim=(-3,-2,-1),norm="backward")
    C=C+f11*f22

print((C-C[:,0:1,...]).abs().max()) # But c is not.
