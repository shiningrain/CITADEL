
import torch
import numpy
import pickle as pkl
np.random.seed(42)
torch.manual_seed(0)

layerinp=torch.Tensor(pkl.load(open("../torch-conv-data.pkl", "rb")))
layerinp=layerinp.to("cuda")

model = torch.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
model.to("cuda")
model.eval()
with torch.no_grad():
    output=model(layerinp).cpu().data.numpy()

print(output[0][0][0])   
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)