
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 50
batch_size = 16
input_w = 224

print(f"torch.__version__: {torch.__version__}")
print(f"device: {device}, steps: {steps}, batch_size: {batch_size}, input_w: {input_w}")

model_grouped: nn.Module = nn.Sequential(
    nn.Conv2d(3, 128, (1, 1)),
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 0
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 1
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 2
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 3
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 4
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 5
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 6
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 7
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 8
    nn.Conv2d(128, 128, (3, 3), padding=1, groups=8),  # 9
    nn.MaxPool2d((input_w, input_w)),
    nn.Flatten(),
    nn.Linear(128, 2),
    nn.Softmax(1))

model_regular: nn.Module = nn.Sequential(
    nn.Conv2d(3, 128, (1, 1)),
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 0
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 1
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 2
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 3
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 4
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 5
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 6
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 7
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 8
    nn.Conv2d(128, 128, (3, 3), padding=1),  # 9
    nn.MaxPool2d((input_w, input_w)),
    nn.Flatten(),
    nn.Linear(128, 2),
    nn.Softmax(1))


def train(label: str, model: nn.Module, data: list[Tuple[torch.Tensor, torch.Tensor]]):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    model.to(device)
    model.train()
    t0 = time.time()
    for (X, y) in data:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{label}: {time.time() - t0:.3f} seconds")


images = torch.from_numpy(np.full((batch_size, 3, input_w, input_w), 0.5)).to(torch.float)
labels = torch.from_numpy(np.full((batch_size, 2), 1)).to(torch.float)
data = [(images, labels)] * 2

train("Warming up GROUPED", model_grouped, data)
train("Warming up REGULAR", model_regular, data)

data = [(images, labels)] * steps
train("Training GROUPED", model_grouped, data)
train("Training REGULAR", model_regular, data)
