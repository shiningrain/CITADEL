import torch
torch.manual_seed(0)
print(f'Running PyTorch version: {torch.__version__}')

torchdevices = [torch.device('cpu')]
# if torch.cuda.is_available():
#   torchdevices.append(torch.device('cuda') )
#   print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))

for torchdevice in torchdevices:
  print('Running on ' + str(torchdevice))

  A_dense = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
          [2.6085, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.1871, 0.0000, 0.0000],
          [0.0000, 0.9922, 0.0000, 0.0000]], device=torchdevice)
  A_csr = A_dense.to_sparse_csr()

  id_csr = torch.sparse_csr_tensor(torch.arange(4+1), torch.arange(4), torch.ones(4, device=torchdevice), (4,4), device=torchdevice)
  print("id_csr:\n",id_csr,"\n",id_csr.to_dense())

  b = torch.ones(4, 2, device=torchdevice)

  print("A_dense:\n",A_dense)
  print("b:\n",b)

  res = torch.triangular_solve(b, A_csr, upper=False, transpose=True, unitriangular=True).solution
  print(res)

  save_path='./tmp_result.pkl'
  with open(save_path, 'wb') as f:
      pickle.dump(r_e_s, f)
