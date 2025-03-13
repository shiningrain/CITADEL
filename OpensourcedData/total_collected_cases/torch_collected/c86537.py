
import random
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, num_layers: int, input_dim: int) -> None:
        super().__init__()
        layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
        )
        self.layers = nn.Sequential(*[
            layer for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

if __name__ == '__main__':
    import os, psutil
    process = psutil.Process(os.getpid())

    times = 0
    while True:
        num_layers = random.randint(1, 15)
        input_dim = random.randint(1, 20)
        m = MyModule(num_layers, input_dim)
        x = torch.randn((1, input_dim))
        exported = torch.jit.trace(m, x)
        o = exported(x)

        times += 1
        if times % 100 == 0:
            mem_mb = process.memory_info().rss / (1024 ** 2)
            print(f'{times} times, mem usage: {mem_mb} MB')
