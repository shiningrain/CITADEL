
import torch
torch.manual_seed(0)
class Anything(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.new_ones((5, ))

    @staticmethod
    def backward(ctx, grad_outputs):

        print("Is gradient w.r.t. sum of b contiguous? ",
              grad_outputs.is_contiguous())
        return grad_outputs.new_zeros((1,))


if __name__ == "__main__":
    funct = Anything.apply
    a = torch.zeros((1,), requires_grad=True)
    b = funct(a)
    bs = b.sum()
    print(str(bs.is_contiguous()))

    save_path='./tmp_result.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(r_e_s, f)