
from torch.nn import functional as F
a = torch.rand(16,64,9,256)
b = torch.rand(16,64,4,256)
print(a.shape)
print(b.shape)
similarities = F.cosine_similarity(a, b, dim = 1)
