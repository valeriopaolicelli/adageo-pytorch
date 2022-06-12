
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class AttenNetVLAD(nn.Module):
    def __init__(self, backbone, netvlad_layer, grl_discriminator=None, attention=False):
        super(AttenNetVLAD, self).__init__()
        self.backbone = backbone
        self.netvlad_layer = netvlad_layer
        self.grl_discriminator = grl_discriminator
        self.weight_softmax = self.backbone.fc.weight
        self.attention = attention
    
    def forward(self, input, grl=False, mode="vlad"):
        if grl:
            _, out, _ = self.backbone(input)
            out = self.grl_discriminator(out)
        else:
            if self.attention:
                fc_out, feature_conv, feature_convNBN = self.backbone(input)
                bz, nc, h, w = feature_conv.size()
                feature_conv_view = feature_conv.view(bz, nc, h * w)
                probs, idxs = fc_out.sort(1, True)
                class_idx = idxs[:, 0]
                scores = self.weight_softmax[class_idx].to(input.device)
                cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
                attention_map = F.softmax(cam.squeeze(1), dim=1)
                attention_map = attention_map.view(attention_map.size(0), 1, h, w)
                attention_features = feature_convNBN * attention_map.expand_as(feature_conv)
                
                if mode == "feat":
                    out = attention_features
                elif mode == "vlad":
                    out = self.netvlad_layer(attention_features)
            else:
                _, out, _ = self.backbone(input)
                if mode == "vlad":
                    out = self.netvlad_layer(out)
        return out


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    
    def __init__(self, num_clusters=64, dim=128, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        
        # Vlad module
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def init_params(self, clsts, traindescs):
        # Init vlad params
        clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending
        
        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clsts_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None
    
    def __vlad_compute_original__(self, x_flatten,soft_assign, N, D):
        vlad = torch.zeros([N, self.num_clusters, D], dtype=x_flatten.dtype, device=x_flatten.device) #24 64 256
        for D in range(self.num_clusters): # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
            vlad[:,D:D+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
    
    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = self.__vlad_compute_original__(x_flatten, soft_assign, N, D)
        return vlad

