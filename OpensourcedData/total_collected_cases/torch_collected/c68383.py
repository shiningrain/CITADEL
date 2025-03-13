py
import torch
from tqdm.auto import tqdm

if __name__ == '__main__':
    x = torch.randn(1).to(0)
    dl = torch.utils.data.DataLoader(
        list(range(5000)),
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    for _ in tqdm(dl):
        pass
    for _ in tqdm(dl):
        pass
