
% python -c "import torch; x=torch.randint(100, (2, 1), dtype=torch.uint8, device='mps');print(x, '\n', x.expand((2, 2)))"
tensor([[45],
        [60]], device='mps:0', dtype=torch.uint8)
tensor([[0, 0],
        [0, 0]], device='mps:0', dtype=torch.uint8)


But the same mysteriously works for int8 tensors:

% python -c "import torch; x=torch.randint(100, (2, 1), dtype=torch.int8, device='mps');print(x, '\n', x.expand((2, 2)))"
tensor([[13],
        [77]], device='mps:0', dtype=torch.int8)
tensor([[13, 13],
        [77, 77]], device='mps:0', dtype=torch.int8)
