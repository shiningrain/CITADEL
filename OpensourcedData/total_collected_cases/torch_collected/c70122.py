
import torch
from kornia.geometry.transform import get_perspective_transform

device = "cuda:0"
# device = "cpu"
dtype = torch.float64

points = [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]

dest = torch.tensor(
    points,
    device=device,
    dtype=dtype,
)

src = torch.tensor(
    points,
    device=device,
    dtype=dtype,
)

# crashes here, should return identity matrix
mat = get_perspective_transform(src, dest)
