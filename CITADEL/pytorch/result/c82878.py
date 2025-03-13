
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import pdb

def pair_muln_selection(src_points, ref_points, log_n_affinity):
    # points feats distance matrix
    ref_dist = torch.cdist(ref_points, ref_points) #(n_t,n_t)
    src_dist = torch.cdist(src_points, src_points) #(n_s,n_s)

    ref_ne_idx = func_cls(5+1, dim=-1, largest=False)[1][:,1:]  #(n_r,a)
    src_ne_idx = func_cls(5+1, dim=-1, largest=False)[1][:,1:]  #(n_s,a)

    src_ne_dist = torch.linalg.norm(src_points[src_ne_idx] - src_points.unsqueeze(1).repeat(1,5,1), dim=-1)  #(n_s,a)
    ref_ne_dist = torch.linalg.norm(ref_points[ref_ne_idx] - ref_points.unsqueeze(1).repeat(1,5,1), dim=-1)  #(n_r,a)

    # pdb.set_trace()
    n_s,n_r = log_n_affinity.size()
    pair_affinity = torch.zeros((n_s,n_r,5,5),device=torch.device('cuda'))  #(n_s,n_r,a,a)
    for i in range(5):
        for j in range(5):
            temp_dist = src_ne_dist[:,i].unsqueeze(-1).repeat(1,n_r) - ref_ne_dist[:,j].unsqueeze(0).repeat(n_s,1)  #(n_s,n_r) 
            temp_dist = temp_dist ** 2 / 0.1 ** 2  #(n_s,n_r)
            temp_dist = torch.clamp(1 - temp_dist, min = 0.)  #(n_s.n_r)
            pair_affinity[:,:,i,j] = log_n_affinity * temp_dist * log_n_affinity[src_ne_idx[:,i]][:,ref_ne_idx[:,j]]


    # _, indices = torch.topk(pair_affinity.cpu().view(-1), 512, dim = -1)
    # indices = indices.to(torch.device('cuda'))

    _, indices = func_cls(pair_affinity.view(-1), 512, dim = -1)
    # indices = indices.to(torch.device('cuda'))

    first_node_src = (indices // (n_r * 5 * 5)).long()  #candidate first src idx
    first_node_ref = ((indices % (n_r * 5 * 5)) // (5 * 5)).long()  #candidate first ref idx
    second_idx = (indices % (n_r * 5 * 5)) % (5 * 5)
    second_node_src = (second_idx // 5).long()  #candidate second src idx
    second_node_ref = second_idx % 5  #candidate second ref idx

    second_node_src = src_ne_idx[first_node_src,second_node_src]
    second_node_ref = ref_ne_idx[first_node_ref,second_node_ref]
    pdb.set_trace()
    return first_node_src,first_node_ref,second_node_src,second_node_ref

if __name__ == '__main__':
    src_points = torch.randn((7000,3),device=torch.device('cuda'))
    ref_points = torch.randn((7000,3),device=torch.device('cuda'))

    log_n_affinity = torch.randn((7000,7000),device=torch.device('cuda'))
    _,_,_,_ = pair_muln_selection(src_points, ref_points, log_n_affinity)
