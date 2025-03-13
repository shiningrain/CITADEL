
  [...]
  File "[...]/test_graph.py", line 785, in test
    loss.backward()
  File "[...]/lib/python3.10/site-packages/torch/_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "[...]/lib/python3.10/site-packages/torch/autograd/__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: unique_by_key: failed to synchronize: cudaErrorStreamCaptureUnsupported: operation not permitted when stream is capturing


For reference, here's a testing script with which I determined the threshold:
py
import torch as th
from torch import nn, optim
from bisect import bisect_left

model = nn.Embedding(5, 30).cuda()
opt = optim.Adam(model.parameters())
inp = (th.arange(0, 10000) % 5).cuda()

def test(N):
    opt.zero_grad(set_to_none=True)
    out = model(inp[:N])
    loss = out.mean()
    loss.backward()
    return None

def capture(N):
    th.cuda.synchronize()
    s = th.cuda.Stream()
    s.wait_stream(th.cuda.current_stream())
    with th.cuda.stream(s):
        for _ in range(3):
            test(N)
    th.cuda.current_stream().wait_stream(s)

    graph = th.cuda.CUDAGraph()
    with th.cuda.graph(graph):
        res = test(N)

def try_capture(N):
    print(f'capture {N}')
    try:
        capture(N)
    except:
        print(f'failed {N}')
        return 2
    print(f'ok {N}')
    return 0

thres = bisect_left(list(range(inp.shape[0])), 1, key=lambda x: try_capture(x))
print(f'>> threshold {thres}')
