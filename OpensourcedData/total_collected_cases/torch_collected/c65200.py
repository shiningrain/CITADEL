
import torch
torch.manual_seed(0)
x = torch.tensor([[[[0.4314, 0.0482, 0.4077, 0.5083, 0.0997, 0.0557, 0.6928],
          [0.9508, 0.5647, 0.9578, 0.7280, 0.3346, 0.6295, 0.7685],
          [0.9874, 0.6847, 0.2015, 0.5326, 0.2211, 0.2419, 0.6180],
          [0.7117, 0.0984, 0.9643, 0.4791, 0.5381, 0.0221, 0.3342]]]], dtype=torch.float32, requires_grad=True, device="cuda")

m = torch.nn.Upsample(scale_factor=1.0239041660002521, mode='bilinear', align_corners=False)
m = m.to("cuda")
y = m(x)
print(y.shape)
y = y.sum()
y.backward()
print(x.grad.cpu().numpy())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)