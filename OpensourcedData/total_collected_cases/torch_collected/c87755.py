
import numpy as np
import torch

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        if data.requires_grad == True:
            return data.detach().cpu().numpy()
        else:
            return data.cpu().numpy()
    else:
        return np.array(data)

def save_model_grad_to_file(model, grad_file):
    grad = {}
    for name, para in model.named_parameters():
        grad[name] = para.grad
    torch.save(grad, grad_file)


def compare_stat_dict_strict(stat_dict_1, stat_dict_2, rtol=5e-2, atol=1e-5):
    for k,v in stat_dict_1.items():
        if k not in stat_dict_2:
            raise KeyError
        else:
            np.testing.assert_allclose(to_numpy(v), to_numpy(stat_dict_2[k]), rtol=rtol, atol=atol)


def check_pytorch_autograd(input_data=None, targets=None, model: torch.nn.Module= None, device='cpu', test_cycle=2, check_model=False):

    #fix input
    if input_data is None:
        np.random.seed(0)
        np_input = np.random.random((32, 3, 32, 32)).astype(np.float32)
        input_data = torch.from_numpy(np_input).to(device)

    # test model
    if model == None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,1),
            torch.nn.BatchNorm2d(32)
        )
    model.train()
    model.to(device)
    loss_fun = torch.nn.MSELoss()

    save_model = False
    for _ in range(test_cycle):
        if save_model == False and not check_model:
            save_model = True

        if save_model:
            torch.save(model.state_dict(), "model_paramter.pth")
        else:
            load_dict = torch.load("model_paramter.pth", map_location=device)
            model.load_state_dict(load_dict, strict=True)

        model_output = model(input_data)

        if save_model:
            np.savez("model_output.npz", out=to_numpy(model_output))
        else:
            test_out = np.load('model_output.npz')['out']
            np.testing.assert_allclose(test_out, to_numpy(model_output))

        loss = loss_fun(torch.ones_like(model_output), model_output)
        print("loss=", loss)
        #empty grad
        for name, para in model.named_parameters():
            para.grad = None
        loss.backward()

        grad = {}
        for name, para in model.named_parameters():
            grad[name] = para.grad
        if save_model:
            torch.save(grad, "model_grad.pth")
        else:
            test_grad = torch.load("model_grad.pth")
            compare_stat_dict_strict(test_grad, grad, rtol=1e-7, atol=0)

    print("check_pytorch_autograd sucess!!!")

if __name__ == "__main__":
   check_pytorch_autograd(check_model=False)    # pytorch 1.11.0 run this will generate the reference grad data
   check_pytorch_autograd(check_model=False)    # pytorch 1.12.1 run


