
import torch

def Build_CoefficientMatrix(c:list,meshx:int,periodic:bool=False):
    '''
    c is list of FD coefficient
    e.g. for 1st derivative with 2nd accuracy central difference:
    c=[-0.5,0] 
    '''
    if 2*len(c)-1>=meshx: raise ValueError
    acc = len(c)   
    
    tmp=[]
    c.reverse()
    for i in range(acc):
        x = torch.cat((torch.cat((torch.zeros((i,meshx-i)),
                                    c[i]*torch.eye(meshx-i)),dim=0),
                                    torch.zeros((meshx,i))
                                    ),dim=1)
        tmp.append(x)
    re=tmp[0]
    for k in tmp[1:]:
        re+=k+k.T

    if periodic:
        re[:acc,-acc:]=re[acc:2*acc,:acc]
        re[-acc:,:acc]=re[:acc,acc:2*acc]
    return re

lapa_c=Build_CoefficientMatrix([1/90,-3/20,3/2,-49/18],64,periodic=True)

x=torch.rand(2,64,1)
out1=torch.matmul(lapa_c,x)
out2=torch.matmul(lapa_c,x[0,:,0])
print(out1[0,0,0].item())
print(out2[0].item())
