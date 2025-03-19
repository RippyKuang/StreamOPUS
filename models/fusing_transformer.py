import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import  FFN
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import  decode_points, encode_points
from .utils import DUMP
from .checkpoint import checkpoint as cp
from .stream import HybridAttention
from .opus_transformer import OPUSSampling,AdaptiveMixing

@TRANSFORMER.register_module()
class FusingTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=1,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[16, 32],
                 scales=[0.5],
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = FusingTransformerDecoder(
            embed_dims, num_layers, num_frames, num_views, num_points, num_levels,
            num_classes, num_groups, num_refines, scales, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, ori_feat, query_points, aligned_query_feat, aligned_query_pos, temp_memory, temp_pos, mlvl_feats, img_metas):
        cls_scores, refine_pts = self.decoder(
            ori_feat, query_points, aligned_query_feat, aligned_query_pos, temp_memory, temp_pos, mlvl_feats, img_metas)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return cls_scores, refine_pts


class FusingTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_layers=1,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[16, 32],
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                FusingTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels, num_classes,
                    num_groups, num_refines[i+1], num_refines[i], layer_idx=i, 
                    scale=scales[i], pc_range=pc_range)
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, ori_feat, query_points, aligned_query_feat, aligned_query_pos, temp_memory, temp_pos, mlvl_feats, img_metas):
        cls_scores, refine_pts = [], []

        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = ori_feat.new_tensor(lidar2img) # [B, N, 4, 4]
        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = ori_feat.new_tensor(ego2lidar) # [B, 4, 4]
        ego2lidar = ego2lidar.unsqueeze(1).expand_as(lidar2img)  # [B, N, 4, 4]
        occ2img = torch.matmul(lidar2img, ego2lidar)

        hybrid_feat = torch.cat([aligned_query_feat,temp_memory],0).permute(1,0,2)
        hybrid_pos = torch.cat([aligned_query_pos,temp_pos],0).permute(1,0,2)
        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            ori_feat, hybrid_feat, cls_score, query_points = decoder_layer(
                ori_feat, query_points, hybrid_feat, hybrid_pos, mlvl_feats, occ2img, img_metas)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        return cls_scores, refine_pts


class FusingTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.layer_idx = layer_idx
        self.scale = scale
        self.d_attn = DecoupledAttention(embed_dims)
        self.h_attn = HybridAttention(embed_dims)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=num_groups, out_points=32)
        self.sampling = OPUSSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                     num_groups=num_groups, num_points=num_points, 
                                     num_levels=num_levels, pc_range=pc_range)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm4 = nn.LayerNorm(embed_dims)
        
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )


        cls_branch = []
        cls_branch.append(nn.Linear(
            self.embed_dims, self.embed_dims))
        for _ in range(num_cls_fcs-1):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        for _ in range(num_reg_fcs-1):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.d_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_feat, query_points, hybrid_feat, hybrid_pos, mlvl_feats, occ2img, img_metas):

        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        hybrid_feat = self.norm2(self.d_attn(hybrid_feat,hybrid_pos))

        hybrid_pos = hybrid_pos.permute(1,0,2)[:1,...]
        hybrid_feat = self.norm3(self.h_attn(query_feat,query_pos,hybrid_feat,hybrid_pos)) # or hybrid_feat=self.ffn(torch.cat(hybrid_feat,query_feat))

        hybrid_feat = self.norm4(self.ffn(hybrid_feat))
        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(hybrid_feat) 
        reg_offset = self.scale * self.reg_branch(hybrid_feat) 
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, hybrid_feat, cls_score, refine_pt

class DecoupledAttention(BaseModule):

    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)


    def inner_forward(self, hybrid_feat,hybrid_pos):
    
        temp_key = temp_value = hybrid_feat
        return self.attention(query = hybrid_feat,
                              key = temp_key,
                              value = temp_value,
                              query_pos=hybrid_pos,
                              key_pos = hybrid_pos
                              ).permute(1,0,2)[:1,...]

    def forward(self, hybrid_feat, hybrid_pos):
        if self.training and hybrid_feat.requires_grad:
            return cp(self.inner_forward, hybrid_feat,hybrid_pos,
                      use_reentrant=False)
        else:
            return self.inner_forward(hybrid_feat,hybrid_pos)
        

