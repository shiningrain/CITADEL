import math
import sys
import pickle
sys.path.append('..')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 25
batch_size = 16
input_w = 224

# print(f"torch.__version__: {torch.__version__}")
# print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
# print(f"torch.backends.cudnn.is_available(): {torch.backends.cudnn.is_available()}")
# print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
# print(f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
# print(f"torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
# print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
# print(f"device: {device}, steps: {steps}, batch_size: {batch_size}, input_w: {input_w}")


def build_model(ch: int, groups: int = 1) -> nn.Module:
    return nn.Sequential(
        func_cls(out_channels=ch, kernel_size=(1,)),
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 0
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 1
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 2
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 3
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 4
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 5
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 6
        func_cls(out_channels=ch, kernel_size=(3,), padding=1, groups=groups),  # 7
        nn.MaxPool2d((input_w, input_w)),
        nn.Flatten(),
        nn.Linear(ch, 2),
        nn.Softmax(1))


def train_model(model: nn.Module, data) -> float:
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
    return time.time() - t0


def perf_test(ch: int, groups: int, dtype) -> float:
    model = build_model(ch, groups).to(dtype)
            images=torch.from_numpy(np.full((batch_size,3,input_w),0.5)).to(dtype)
    labels = torch.from_numpy(np.full((batch_size, 2), 1)).to(dtype)
    train_model(model, [(images, labels)] * 2)  # Warmup
    return train_model(model, [(images, labels)] * steps)


# print(f"dtype\tch\tgroups\tseconds\texpected\tratio\tanalysis\tdt0_ratio")
for dtype in [torch.float16]:#torch.float32, 
    dtype_s = str(dtype).replace("torch.", "")
    dt0_last = float("nan")
    for ch0 in [64]:#, 1024
        dt0 = perf_test(ch0, 1, dtype)
        dt0_ratio = f"{dt0 / dt0_last:.3f}" if not math.isnan(dt0_last) else ""
        dt0_last = dt0
        for groups in [4]:
            ch = ch0 * groups
            if ch > 1024:  # Takes forever
                continue
            dt = perf_test(ch, groups, dtype)
            expected = dt0 * groups
            ratio = dt / expected
        r_e_s=ratio
        save_path='./tmp_result.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(r_e_s, f)