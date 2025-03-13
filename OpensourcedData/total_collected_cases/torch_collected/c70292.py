
import torch
arg_1 = torch.tensor(
        [[[0.7765],
          [0.6088],
          [0.3014]],

         [[0.7765],
          [0.3868],
          [0.0320]]])
arg_2 = torch.tensor(
        [[[0.4053, 0.6713, 0.8894],
          [0.7145, 0.0562, 0.0616],
          [0.6813, 0.5486, 0.9225]],

         [[0.8425, 0.3334, 0.1288],
          [0.3434, 0.7524, 0.9728],
          [0.4400, 0.3666, 0.7364]]])
arg_3 = torch.tensor([[0, 7, 7],
        [4, 3, 6]], dtype=torch.int32)
res= torch.lu_solve(arg_1,arg_2,arg_3,)
# double free or corruption (out)
# [1]    7940 abort (core dumped)  python a.py
