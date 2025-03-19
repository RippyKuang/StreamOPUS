import torch
import torch.nn as nn
import numpy as np
from .bbox.utils import decode_points
from .positional_encoding import *
from .checkpoint import checkpoint as cp
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import MultiheadAttention

def topk_gather(feat, topk_indexes):
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
        return feat
    
def memory_refresh(memory, prev_exist):
        memory_shape = memory.shape
        view_shape = [1 for _ in range(len(memory_shape))]
        prev_exist = prev_exist.view(-1, *view_shape[1:]) 
        return memory * prev_exist

class Memory:

    def __init__(self, topk_proposals, num_propagated, memory_len, num_query, pc_range, embed_dims, num_points,with_ego_pos=True):
        self.topk_proposals = topk_proposals 
        self.num_propagated = num_propagated 
        self.memory_len = memory_len 
        self.num_query = num_query
        self.pc_range =torch.Tensor(pc_range).cuda()
        self.embed_dims = embed_dims
        self.num_points = num_points 
        self.with_ego_pos =with_ego_pos

        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Parameter(torch.Tensor(self.num_propagated , num_points, 3)).cuda().detach()

        
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        ).cuda()

        self.spatial_alignment = MLN(8).cuda()

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        ).cuda()

        self.ego_pose_pe = MLN(180).cuda()
        self.ego_pose_memory = MLN(180).cuda()

        self.reset_memory()
        self.init_weight()
    
    def init_weight(self):
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.data, 0, 1)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None

    def pre_update_memory(self, metas):
        x = metas[0]['prev_exists']
        x = torch.tensor(x,device='cuda').int()

        B = len(metas)

        batched_timestamp,batched_inv_pose = [],[]
        for i in range(B):
            batched_timestamp.append(torch.tensor([metas[i]['img_timestamp'][0]],device='cuda')*1e-6)
            batched_inv_pose.append(torch.tensor(np.linalg.inv(metas[i]['lidar2global']),device='cuda',dtype=torch.float32))
        batched_timestamp = torch.stack(batched_timestamp,0).unsqueeze(1)
        batched_inv_pose  = torch.stack(batched_inv_pose,0).unsqueeze(1)

        if self.memory_embedding is None:
            self.memory_embedding = torch.zeros(B, self.memory_len, self.embed_dims, device='cuda')
            self.memory_reference_point = torch.zeros(B, self.memory_len, self.num_points, 3, device='cuda')
            self.memory_timestamp = torch.zeros(B, self.memory_len, 1, device='cuda')
            self.memory_egopose = torch.zeros(B, self.memory_len, 4, 4, device='cuda')
        else:
            self.memory_timestamp += batched_timestamp
            self.memory_egopose = batched_inv_pose @ self.memory_egopose
            self.memory_reference_point =  torch.matmul(batched_inv_pose[...,:3,:3].reshape(1,1,1,3,3), \
                            self.memory_reference_point.unsqueeze(-1)).squeeze(-1)+batched_inv_pose[...,:3,3]
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
        
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device='cuda')

    def propagate(self, tgt, reference_points):

        temp_reference_point = ((self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])).mean(-2)
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding

        if self.with_ego_pos:
            memory_ego_motion = torch.cat([temp_reference_point, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated].unsqueeze(-2)], dim=1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion[:, self.num_propagated:])

        return tgt, reference_points, temp_memory, temp_pos


    def temporal_alignment(self, tgt, reference_points):  # tgt: current query
        center_points = reference_points.mean(-2)
        query_pos = self.query_embedding(pos2posemb3d(center_points))
        B = query_pos.size(0)

        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([center_points, rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            aligned_query_feat = self.ego_pose_memory(tgt, rec_ego_motion) 
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
         
        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(center_points[...,:1])))
        
        return aligned_query_feat,query_pos


    def post_update_memory(self, metas, feats, all_scores, pts):
        B, _, _ = feats.shape
        all_pts,all_feat,batched_pose = [],[],[]
        for i in range(B):
            scores, _= all_scores.sigmoid().max(-1)
            _, topk_indexes = torch.topk(scores.mean(-1), self.topk_proposals, dim=-1)
            all_feat.append(topk_gather(feats,topk_indexes).detach())
            all_pts.append(decode_points(topk_gather(pts,topk_indexes).detach(),self.pc_range))
            batched_pose.append(pts.new_tensor(metas[i]['lidar2global']))

        batched_pose  = torch.stack(batched_pose,0).unsqueeze(1)
        self.memory_timestamp = torch.cat([torch.zeros([B,self.topk_proposals,1], device='cuda'),self.memory_timestamp],1)
        self.memory_embedding = torch.cat([torch.cat(all_feat,0),self.memory_embedding],1)
        self.memory_reference_point = torch.cat([torch.cat(all_pts,0),self.memory_reference_point],1)
        self.memory_reference_point =  torch.matmul(batched_pose[...,:3,:3].reshape(1,1,1,3,3), \
                                            self.memory_reference_point.unsqueeze(-1)).squeeze(-1)+batched_pose[...,:3,3]
        self.memory_egopose = torch.cat([torch.eye(4,device='cuda').reshape(1,1,4,4).repeat(1,self.topk_proposals,1,1),self.memory_egopose],1)

        batched_timestamp,batched_pose = [],[]
        for i in range(B):
            batched_timestamp.append(torch.tensor([metas[i]['img_timestamp'][0]],device='cuda')*1e-6)
            batched_pose.append(torch.tensor(metas[i]['lidar2global'],device='cuda',dtype=torch.float32))
        batched_timestamp = torch.stack(batched_timestamp,0).unsqueeze(1)
        batched_pose  = torch.stack(batched_pose,0).unsqueeze(1)

        self.memory_timestamp -= batched_timestamp
        self.memory_egopose = batched_pose @ self.memory_egopose

class MLN(nn.Module):

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


class HybridAttention(BaseModule):

    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)


    def inner_forward(self, query, query_pos, temp_memory,temp_pos):
        temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
        temp_pos = torch.cat([query_pos, temp_pos], dim=1)
        return self.attention(query = query,
                              key = temp_key,
                              value = temp_value,
                              query_pos=query_pos,
                              key_pos = temp_pos
                              )

    def forward(self, query, query_pos, temp_memory, temp_pos):
        if self.training and query.requires_grad:
            return cp(self.inner_forward, query, query_pos, temp_memory, temp_pos,
                      use_reentrant=False)
        else:
            return self.inner_forward(query, query_pos, temp_memory, temp_pos)
        

        
