
import torch
A = torch.Tensor(
      [[ 8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  9.,  1.],
        [10.,  9.,  8.,  7.,  6.,  5.,  4.,  3., 11.,  1.],
        [11., 10.,  9.,  8.,  7.,  6.,  5.,  4., 12.,  1.],
        [12., 11., 10.,  9.,  8.,  7.,  6.,  5., 13.,  1.],
        [13., 12., 11., 10.,  9.,  8.,  7.,  6., 14.,  1.],
        [14., 13., 12., 11., 10.,  9.,  8.,  7., 15.,  1.],
        [15., 14., 13., 12., 11., 10.,  9.,  8., 16.,  1.],
        [16., 15., 14., 13., 12., 11., 10.,  9., 17.,  1.],
        [17., 16., 15., 14., 13., 12., 11., 10., 18.,  1.],
        [18., 17., 16., 15., 14., 13., 12., 11., 19.,  1.]]
)
# assert A.det() == 0
# print(A.inverse()) # <-- No error!

Ac = A.cuda()
print(torch.inverse(Ac)) # <-- _LinAlgError: linalg.inv: (Batch element 0): The diagonal element 9 is zero, the inversion could not be completed because the input matrix is singular.
