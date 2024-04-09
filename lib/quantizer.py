import torch.nn as nn
import math
import torch
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import random
import math
from tqdm.auto import trange
# import ipynbname  # pip install ipynbname

import torch.nn as nn
import torch.nn.functional as F
import transformers
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from typing import Optional,Union,List
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 100,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param greedy_init: if True, init by greedily selecting the point that is farthest from any cluster
        if False (default), initialize with random points using pytorch global RNG
    :param block_size_vals: how many dot products to compute at a time
    :param devices: if specified, run kmeans in data-parallel mode across these devices
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if devices is None:
        devices = [data.device]

    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]

    block_size = block_size_vals // k
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for i in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                    torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                    data[gi][block_start : block_start + block_size],
                    clusters[gi].T,
                    beta=-0.5,
                ).argmax(1)
            # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        if len(devices) == 1:
            new_clusters = [
                clusters[0]
                .clone()
                .index_reduce_(dim=0, index=nearest_indices[0], source=data[0], reduce="mean", include_self=False)
            ]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest_indices[gi], source=data[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest_indices[gi], minlength=k).to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]

            new_clusters = [cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)]
            new_clusters[0] += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            for gi in range(1, len(devices)):
                new_clusters.append(new_clusters[0].to(devices[gi], non_blocking=True))

        if i % check_every == 0:
            if torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data

def get_nearest_indices(
    S: torch.Tensor, #重要性
    W,
    shape, # 权重的原始形状
    centroids,
    devices: Optional[List[torch.device]] = None,
):
    if S is None:
        S = torch.zeros(shape[0]).to(W.device)
        S[0] = 1
        # S[0] = 1
    # if devices is None:
    #     devices = [data.device]
    # W  N*D
    # centroids n_centroids*D
    assignments_list = []
    a1 = W.view(-1,centroids.shape[-1]).unsqueeze(1)
    # S为每一行的重要性权重，将其扩展成矩阵形式，方便计算
    s1 = S.view(-1,centroids.shape[-1]).unsqueeze(1)
    chunks_a = torch.chunk(a1, 5, dim=0)
    chunks_s = torch.chunk(s1, 5, dim=0)
    b1 = centroids.unsqueeze(0)
    for ac,sc in zip(chunks_a,chunks_s):
        dist = ((ac-b1)**2*sc).sum(-1)
        assignments_list.append(dist.argmin(-1))
    
    assignments = torch.cat(assignments_list,dim=0)
    # dist =((a1-b1)**2*s1).sum(-1)
    # assignmentss = dist.argmin(-1)
    # print(assignments - assignmentss)
    # print(assignments.shape)
    return assignments
def quantize(org_weight,codebook_num = 2,centroids_num = 256,block_size = 64,centroid_len = 8):
    # 计算每一行的二范数
    # max_matrix = get_max(org_weight)
    reshspe_weight = org_weight.view(-1,block_size)
    scales = reshspe_weight.norm(p=2, dim=1, keepdim=True).float()
    # nn.Parameter(scales, requires_grad=True)
    # 每一行除以其对应的范数
    normalized_tensor = (reshspe_weight / scales)
    weight_list = normalized_tensor.split(normalized_tensor.shape[0]//codebook_num,dim = 0)
    clusters_list = []
    nearest_indices_list = []
    # reconstructed_data_list = []
    for weight in weight_list:
        clusters, nearest_indices, _=fit_kmeans(weight.view(-1,centroid_len),k = centroids_num,max_iter= 100)
        clusters_list.append(clusters.unsqueeze(0))
        nearest_indices_list.append(nearest_indices.unsqueeze(0))
        # reconstructed_data_list.append(reconstructed_data.view(weight.shape))
    clusters_merge = torch.cat(clusters_list,dim = 0)
    nearest_indices_merge = torch.cat(nearest_indices_list,dim = 0)
    # reconstructed_data_merge = (torch.cat(reconstructed_data_list,dim = 0)*scales)
    # reconstructed_data_merge = reconstructed_data_merge.view(org_weight.shape)
    return clusters_merge,nearest_indices_merge,scales
def col_wise_class(org_weight,class_num,max_iter = 100):
    clusters, nearest_indices, reconstructed_data=fit_kmeans(org_weight,k = class_num,max_iter= 500)
    return nearest_indices
class Quantization(nn.Module):
    
    def __init__(self,layer,codebook_num = 2,centroids_num = 256,bolck_size = None,centroid_len = 4) -> None:
        super().__init__()
        self.layer = layer
        self.dev = self.layer.weight.device 
        W = self.layer.weight.data.float().clone().cuda()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev,dtype= torch.float32)
        self.nsamples = 0
        self.codebook_num = codebook_num
        self.centroids_num = centroids_num
        self.centroid_len = centroid_len
        if bolck_size is None:
            bolck_size = W.shape[0]
        clusters_merge,nearest_indices_merge,scales \
            = quantize(W.float(),codebook_num=codebook_num,block_size=bolck_size,centroid_len=centroid_len)
        self.codebooks = nn.Parameter(clusters_merge,requires_grad=True)
        self.scales = nn.Parameter(scales,requires_grad=True)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.codes = nn.Parameter(nearest_indices_merge,requires_grad=False)
        # self.reconstructed_data_merge = reconstructed_data_merge.to(self.dev)
        self.bolck_size=bolck_size
    def add_batch(self, inp, out):
        # print("add_batch:inp{},layer",inp.shape,self.layer.weight.shape)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples 
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += inp.abs().mean(dim=1)
        # inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        self.H /=self.nsamples
    def differentiable_dequantize(self):
        codebook_num = self.codebook_num
        codes = self.codes.clone().detach()
        for i in range(codebook_num):
            codes[i,:]+=self.centroids_num*i
        # _,count =  torch.unique(codes, return_counts=True)
        # print(count.min(),count.max)
        codebook_offsets = torch.arange(0,self.layer.weight.data.numel()//self.centroid_len).to(self.dev)
        reconstruct_weight = F.embedding_bag(codes.flatten(),self.codebooks.flatten(0,1),codebook_offsets,mode="sum")
        return (reconstruct_weight.view(-1,self.bolck_size)*self.scales).view((self.rows, self.columns))
    def update_index(self,residual,num = 100):
        reshspe_weight = self.layer.weight.data.clone().to(residual.device)
        if residual is not None:
            reshspe_weight = reshspe_weight - residual
        reshspe_weight = reshspe_weight.view(-1,self.bolck_size)
        detach_scales = self.scales.detach()
        normalized_tensor = (reshspe_weight / detach_scales)
        S = self.scaler_row.unsqueeze(0)\
            .expand(self.layer.weight.data.shape[0], -1)\
                .contiguous()\
                    .view(-1,self.bolck_size)
        weight_list = normalized_tensor.split(normalized_tensor.shape[0]//self.codebook_num,dim = 0)
        S_list = S.split(normalized_tensor.shape[0]//self.codebook_num,dim = 0)
        nearest_indices_list = []
        index = 0
        for weight,s in zip(weight_list,S_list):
            # search
            nearest_indices=get_nearest_indices(S=s,W = weight.view(-1,self.centroid_len),shape = weight.shape,centroids=self.codebooks[index])
            nearest_indices_list.append(nearest_indices.unsqueeze(0))
            index +=1
        del S
        nearest_indices_merge = torch.cat(nearest_indices_list,dim = 0)
        self.codes.data  =nearest_indices_merge
    def forward(self):
        weight = self.differentiable_dequantize()
        return weight