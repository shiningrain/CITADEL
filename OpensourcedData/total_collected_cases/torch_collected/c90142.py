
import torch

# N C L
# print("✔️ No padding")
# print(f"{torch.nn.Conv1d(1, 1, 1, padding=0)(torch.ones(16, 1, 1)).shape = }")
# print("✔️ Padding but let the batch size < 16 is okay.")
# print(f"{torch.nn.Conv1d(1, 1, 1, padding=1)(torch.ones(15, 1, 1)).shape = }")
print("❌ Padding and let the batch size > 16 is **NOT** okay.")
print(f"{torch.nn.Conv1d(1, 1, 1, padding=1)(torch.ones(16, 1, 1)).shape = }")
