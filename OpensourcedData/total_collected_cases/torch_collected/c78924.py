
import torch
import itertools

def _samples_to_XY(samples):
    X = [x for x, _ in samples]
    X = torch.tensor(X)
    Y = torch.tensor([y for _, y in samples])
    return X, Y

def _gen(tensor_size):
    while True:
        yield (list(range(tensor_size)), 1)

def _gen_batch(tensor_size, batch_size):
    gen = _gen(tensor_size)
    while True:
        xys = list(itertools.islice(gen, batch_size))
        if not xys:
            break
        yield _samples_to_XY(xys)

class TheDataset(torch.utils.data.IterableDataset):
    def __init__(self, tensor_size, batch_size):
        self.batch_size = batch_size
        self.tensor_size = tensor_size
    def __iter__(self):
        return _gen_batch(self.tensor_size, self.batch_size)
    def into_dataloader(self, num_workers):
        mproc_config = {'multiprocessing_context': 'fork'}
        mproc_config = mproc_config if num_workers > 0 else {}
        return torch.utils.data.DataLoader(self,
                                           num_workers=num_workers,
                                           **mproc_config)

if __name__ == '__main__':

    some_tensor = torch.rand(10000, 50)
    
    # if this statement is commented, none of the code below hangs
    some_tensor = some_tensor.clone()

    # works ok
    if next(iter(TheDataset(tensor_size=64, batch_size=512).into_dataloader(num_workers=1))):
        print('ok')

    # works ok
    if next(iter(TheDataset(tensor_size=128, batch_size=256).into_dataloader(num_workers=1))):
        print('ok')

    # hangs
    if next(iter(TheDataset(tensor_size=128, batch_size=512).into_dataloader(num_workers=1))):
        print('ok')
