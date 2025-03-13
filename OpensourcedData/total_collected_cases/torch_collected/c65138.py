
#%debug
import torch
import torchvision
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as Seq, ReLU, AvgPool1d, Dropout, Softmax, Sigmoid, BatchNorm1d
from torch_geometric.nn import EdgeConv, DynamicEdgeConv, BatchNorm, avg_pool_x, knn_graph, EdgePooling, DenseSAGEConv, dense_diff_pool, dense_mincut_pool, SAGEConv, max_pool_x, SAGPooling, JumpingKnowledge, global_mean_pool as gap, global_max_pool as gmp
#from torch_cluster import knn_graph

#dummy_input = (torch.randn(10,5, device='cuda'),torch.randn(2,10, device='cuda').type(torch.int64))
coords0 = torch.randn(1,6)
coords1 = torch.randn(1,6)


coords = torch.transpose(torch.cat((coords0,coords1),dim=0),0,1)
adj = knn_graph(coords, k=2, batch=None, loop=True)

edge_from = adj[0:1,:]
edge_to = adj[1:,:]

input_values = (coords0,coords1, edge_from, edge_to)
#model = torchvision.models.alexnet(pretrained=True).cuda()
class trial1(torch.nn.Module):
    def __init__(self):
        super(trial1, self).__init__()
        
        self.SAGEConvBlock1 = SAGEConv(2, 512, normalize=True)
        self.bano1 = BatchNorm(512)
        self.relu = ReLU()
        self.dense1 = Seq(Lin(512,1))
        self.sigmoid = Sigmoid()
    def forward(self, coords0, coords1, edge_from, edge_to):
      adj = torch.cat((edge_from,edge_to),dim=0)
      gra = torch.transpose(torch.cat((coords0,coords1),dim=0),0,1)
      
      x1 = self.SAGEConvBlock1(gra, edge_index=adj)
      x = torch.unsqueeze(torch.sum(x1), dim=0)
      return x

model2 = trial1()

input_names = ["coords0","coords1", "edge_from", "edge_to"]
output_names = [ "outputs" ]

torch.onnx.export(model2, input_values, "ONNX_models/trial2.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11, dynamic_axes={'coords0':{0:'batch_size',1:'features'}, 'coords1':{0:'batch_size',1:'features'}, 'edge_from':{0:'batch_size',1:'features'}, 'edge_to':{0:'batch_size',1:'features'}, 'outputs':{0:'batch_size'}})


The onnx runtime code to perform the inference is:

import onnxruntime
print(onnxruntime.__version__)
ort_session = onnxruntime.InferenceSession("ONNX_models/trial2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(coords0), ort_session.get_inputs()[1].name: to_numpy(coords1), ort_session.get_inputs()[2].name: to_numpy(edge_from), ort_session.get_inputs()[3].name: to_numpy(edge_to)}
ort_outs = ort_session.run(["outputs"], ort_inputs)
