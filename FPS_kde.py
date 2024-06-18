import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from pointnet2_ops import pointnet2_utils
import math

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

def fps_downsample(pc, x, num_group):
    """
    输入参数：
        pc: 坐标点，形状为 (batch_size, num_points, 3)，表示每个点的三维坐标。
        x: 特征向量，形状为 (batch_size, num_points, channels)，表示每个点的特征向量。
        num_group: 采样后的点数。
    
    输出：
        new_coor: 采样后的坐标点，形状为 (batch_size, num_group, 3)。
        new_x: 采样后的特征向量，形状为 (batch_size, num_group, channels)。
    """   
    fps_idx = pointnet2_utils.furthest_point_sample(pc.contiguous(), num_group)

    combined_x = torch.cat([pc, x], dim=2)
    new_combined_x = (
        pointnet2_utils.gather_operation(
            combined_x.transpose(1, 2).contiguous(), fps_idx
        )
    )
    #根据采样点的索引，从拼接后的张量中提取采样后的坐标点 new_coor 和特征向量 new_x。
    new_combined_x = new_combined_x.transpose(1, 2).contiguous()
    new_pc = new_combined_x[:, :, :3]
    new_x = new_combined_x[:, :, 3:]

    return new_pc, new_x

def kernel_density_estimation_ball(pts, radius, sigma, nsample=128, is_norm=False):
    #points pts [B,N,3]
    #idx(b, N, nsample=128), pts_cnt (b,N)
    idx, pts_cnt = pointnet2_utils.ball_query(radius, nsample, pts, pts)
    g_pts = index_points(pts, idx.long()) # [B, N, nsample, C]
    g_pts -= pts.unsqueeze(2).expand_as(g_pts)
    
    R = math.sqrt(sigma)
    xRinv = g_pts / R
    quadform = (xRinv ** 2).sum(dim=-1)
    logsqrtdetSigma = math.log(R) * 3
    mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - 3 * math.log(2 * 3.1415926) / 2)
    
    first_val, _ = torch.split(mvnpdf, [1, nsample - 1], dim=2)
    mvnpdf = mvnpdf.sum(dim=2, keepdim=True)
    num_val_to_sub = (nsample - pts_cnt).float().unsqueeze(-1)
    val_to_sub = first_val * num_val_to_sub
    mvnpdf -= val_to_sub

    scale = 1.0 / pts_cnt.float().unsqueeze(-1)
    density = mvnpdf * scale

    if is_norm:
        density_max = density.max(dim=1, keepdim=True)[0]
        density /= density_max
    return density

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((2,2048,3))
    new_xyz = fps(input, npoint=16)
    xyz_density = kernel_density_estimation_ball(xyz, sigma=0.1, sigma=0.1)
  
