
import multiprocessing
import torch
import torch.nn as nn
from torch.nn import functional as F

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # no need for all this to have the error
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        pass

def init_model_and_load():
    model = TheModelClass()
    print("before load")
    model.load_state_dict(torch.load("./model.torch", map_location=torch.device('cpu')))
    print("after load") # never reaches it
    return

if __name__ == "__main__":
    # First, let's save a random init model
    model = TheModelClass()
    torch.save(model.state_dict(), "./model.torch")
    print("nb params", sum(p.numel() for p in model.parameters()))
    # it's probably related to the number of params : if we remove model.fc1, it works

    size = [33**3] # 32**3 works, but not 33**3 ... :scream:
    tensor = torch.rand(size) 
    pad = (0, 0)
    # whether you want the error or not...
    I_WANT_THIS_NOT_TO_WORK = True
    if I_WANT_THIS_NOT_TO_WORK:
        tensor = F.pad( # everything about this bug comes from padding, even with 0
            tensor,
            pad,
            mode="constant",
            value=0,
        )
    # and it doesn't work, even if you reinitialize the tensor!
    tensor = torch.rand([*size])
    p = multiprocessing.Process( # no need to pass tensor as argument, it still fails
        target=init_model_and_load,
    )
    p.start()
    p.join()
