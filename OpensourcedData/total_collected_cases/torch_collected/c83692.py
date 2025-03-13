
% python -c "import torch;print(torch.full((2, 2), 247, device='mps', dtype=torch.uint8))"
tensor([[0, 0],
        [0, 0]], device='mps:0', dtype=torch.uint8)


But the same works for values smaller than 128:

% python -c "import torch;print(torch.full((2, 2), 127, device='mps', dtype=torch.uint8))"
tensor([[127, 127],
        [127, 127]], device='mps:0', dtype=torch.uint8)
