
import torch
results = dict()
row = 0
col = 1
offset = 2
try:
  results["res_1"] = torch.tril_indices(row, col, offset=offset)
except Exception as e:
  results["err_1"] = str(e)
try:
  results["res_2"] = torch.triu_indices(row,col,offset=offset)
except Exception as e:
  results["err_2"] = str(e)

print(results)


By contrast, when `row=1` and `col=0`, they both return empty tensor which I think is correct.


import torch
results = dict()
row = 1
col = 0
offset = 2
try:
  results["res_1"] = torch.tril_indices(row, col, offset=offset)
except Exception as e:
  results["err_1"] = "ERROR:"+str(e)
try:
  results["res_2"] = torch.triu_indices(row,col,offset=offset)
except Exception as e:
  results["err_2"] = "ERROR:"+str(e)

print(results)
# {'res_1': tensor([], size=(2, 0), dtype=torch.int64), 'res_2': tensor([], size=(2, 0), dtype=torch.int64)}
