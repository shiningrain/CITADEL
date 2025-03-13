
from math import pi

import torch
from torch.fft import fft2, ifft2, fftshift
from skimage import io

# Pixel Pitch
δ = (8.0e-6, 8.0e-6)

# Wavelength
λ = 515e-9

# Distance
dist = 0.1

# Image
image = 'Holo.png'

# Main
if __name__ == '__main__':
        
    # Read Image
    target = io.imread(image, as_gray=True)

    # Image To 3D
    target = target / 255.0
    
    # Image To Tensor
    target = torch.tensor(target, dtype=torch.double, device='cuda:0')
    
    # Phase
    target = torch.ones_like(target) * torch.exp(2j * pi * target)
    
    # Index X
    indexX = torch.tensor([[i for i in range(target.shape[-1])] for _ in range(target.shape[-2])], dtype=torch.double, device='cuda:0')
    
    # Index Y
    indexY = torch.tensor([[i for _ in range(target.shape[-1])] for i in range(target.shape[-2])], dtype=torch.double, device='cuda:0')
    
    # Coordinate X
    coordX = indexX - target.shape[-1] / 2 + 0.5
    
    # Coordinate Y
    coordY = indexY - target.shape[-2] / 2 + 0.5
    
    # Fx
    diff_X = coordX / (δ[-1] * target.shape[-1])
    
    # Fy
    diff_Y = coordY / (δ[-2] * target.shape[-2])
    
    # Phase
    phi = 2 * pi * dist / λ * torch.sqrt(1 - (λ ** 2) * (diff_X ** 2 + diff_Y ** 2))
    
    # Window
    window = torch.exp(1j * phi)
    
    # Propagation #1
    middle = fftshift( fft2(fftshift(target)))
    
    # Propagation #2
    middle = middle * window
    
    # Propagation #3
    result = fftshift(ifft2(fftshift(middle)))
    
    # Result
    result = torch.abs(result) / torch.max(torch.abs(result))
    
    # Save
    io.imsave('Reconstruct.png', (result * 255).to(torch.uint8).squeeze().detach().cpu().numpy())
