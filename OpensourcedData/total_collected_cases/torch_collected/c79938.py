

import torch

feature_extractor = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(8, 16, kernel_size=4),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3)
)

b_img = torch.rand(32, 3, 28, 28, device='meta')
feature_extractor.to('meta')

print(feature_extractor(b_img).shape)


is not possible while



import torch

feature_extractor = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, kernel_size=3),
    torch.nn.Sigmoid(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(8, 16, kernel_size=4),
    torch.nn.Sigmoid(),
    torch.nn.MaxPool2d(kernel_size=3)
)

b_img = torch.rand(32, 3, 28, 28, device='meta')
feature_extractor.to('meta')

print(feature_extractor(b_img).shape)
