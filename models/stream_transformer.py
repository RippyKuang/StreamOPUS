import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.utils.builder import TRANSFORMER
from .stream import HybridAttention

@TRANSFORMER.register_module()
class StreamTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_layers=6, 
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.decoder = StreamTransformerDecoder(embed_dims, num_layers)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, hybrid_query_feat, query_pos, temp_memory, temp_pos):
        out_feat = self.decoder(hybrid_query_feat, query_pos, temp_memory, temp_pos)
        return out_feat


class StreamTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_layers=6,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
      
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                StreamTransformerDecoderLayer(embed_dims)
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_feat, query_pos, temp_memory, temp_pos):
        out_feat = query_feat
        for i, decoder_layer in enumerate(self.decoder_layers):
            out_feat = decoder_layer(out_feat, query_pos, temp_memory, temp_pos)
            
        return out_feat


class StreamTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.hybrid_attn = HybridAttention(embed_dims, num_heads=8, dropout=0.1)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query_feat, query_pos, temp_memory, temp_pos):

        query_feat = self.hybrid_attn(query_feat, query_pos, temp_memory, temp_pos)
        query_feat = self.norm(self.ffn(query_feat))

        return query_feat
