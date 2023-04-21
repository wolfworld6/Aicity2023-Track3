import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, PoolFormerEmbeddings, PoolFormerLayer, MaskedConv1D,
                     ConvBlock, LayerNorm, SGPBlock)
import pdb

@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch,
        channel_att_sride = 2,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.arch = arch                                                          # (2, 2, 5)
        self.mha_win_size = mha_win_size                                          # [9, 9, 9, 9, 9, 9]
        self.max_len = max_len                                                    # 2304
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor                                          # 2
        self.use_abs_pe = use_abs_pe                                              # false
        self.use_rel_pe = use_rel_pe                                              # false
        # pdb.set_trace()
        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)                # n_embd: 512
        if self.use_abs_pe:                                                       # train & val: False
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        
        # MaskedConv1D:
            # (conv0): Conv1d(2304, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
            # (conv1): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        for idx in range(arch[0]):                                                # 2
            if idx == 0:
                in_channels = n_in                                                # 2304
                # self.embd.append(MaskedConv1D(
                #     in_channels, n_embd, 7,                                       # 7
                #     stride=1, padding=n_embd_ks//2 + 2, bias=(not with_ln)            # 3
                #     )
                # )
                # self.embd.append(MaskedConv1D(
                #         n_embd, in_channels, n_embd_ks,                               # 3
                #         stride=1, padding=n_embd_ks//2, bias=(not with_ln)            # 1
                #     )
                # )
                self.embd.append(MaskedConv1D(
                        in_channels, n_embd, n_embd_ks,                               # 3
                        stride=1, padding=n_embd_ks//2, bias=(not with_ln)            # 1
                    )
                )
            else:
                in_channels = n_embd                                              # 512
            # self.embd.append(MaskedConv1D(
            #         in_channels, n_embd, 7,                                       # 7
            #         stride=1, padding=n_embd_ks//2 + 2, bias=(not with_ln)            # 3
            #     )
            # )
            # self.embd.append(MaskedConv1D(
            #         n_embd, in_channels, n_embd_ks,                               # 3
            #         stride=1, padding=n_embd_ks//2, bias=(not with_ln)            # 1
            #     )
            # )
                self.embd.append(MaskedConv1D(
                        in_channels, n_embd, n_embd_ks,                               # 3
                        stride=1, padding=n_embd_ks//2, bias=(not with_ln)            # 1
                    )
                )
            # self.embd.append(PoolFormerEmbeddings(
            #         patch_size=3,
            #         stride=1,
            #         padding=n_embd_ks//2,
            #         num_channels=in_channels,
            #         hidden_size=512
            #     )
            # )
            # pdb.set_trace()
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))                          # true
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(
                    n_embd, n_head,                                               # 4
                    n_ds_strides=(1, 1),
                    channel_att_sride=self.scale_factor ** idx,
                    attn_pdrop=attn_pdrop,                                        # 0
                    proj_pdrop=proj_pdrop,                                        # 0.1
                    path_pdrop=path_pdrop,                                        # 0.1
                    mha_win_size=self.mha_win_size[0],                            # 9
                    use_rel_pe=self.use_rel_pe                                    # false
                )
            )
        # pdb.set_trace()
        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            # print("----------------------------------- channel_att_sride ",self.scale_factor ** idx)
            self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    channel_att_sride=self.scale_factor ** idx,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1+idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()                                                      # x:[1, 2304, 2304] mask:[1, 1, 2304]
        # print("len(self.embd): ",len(self.embd))
        # embedding network
        for idx in range(len(self.embd)):                                       # 2
            x, mask = self.embd[idx](x, mask)                                      # x0:[1, 512, 2304] mask0:[1, 1, 2304]    # x1:[1, 512, 2304] mask1:[1, 1, 2304]
            if idx > 1:
                x = self.relu(self.embd_norm[idx-2](x))

        # pdb.set_trace()
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:                                   # test mode
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.float()

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.float()

        # stem transformer
        for idx in range(len(self.stem)):                                        # 2       
            x, mask = self.stem[idx](x, mask)                                       # x0:[1, 512, 2304] mask0:[1, 1, 2304]    # x1:[1, 512, 2304] mask1:[1, 1, 2304]
        
        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x, )                                                       # [1, 512, 2304]
        out_masks += (mask, )                                                    # [1, 1, 2304]

        # main branch with downsampling
        for idx in range(len(self.branch)):                                      # 5
            x, mask = self.branch[idx](x, mask)                                     # 0:[1, 512, 1152] [1, 1, 1152] 1:[1, 512, 576] [1, 1, 576]
            out_feats += (x, )                                                      # [1, 512, 2304]                                                     
            out_masks += (mask, )                                                   # [1, 1, 2304]

        return out_feats, out_masks                                              # 5
                                                                                    # out_feats 0:[1, 512, 2304] 1:[1, 512, 1152] 2:[1, 512, 576] 3:[1, 512, 288] 4:[1, 512, 144] 5:[1, 512, 72]
                                                                                    # out_masks 0:
@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                    in_channels, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x, )
        out_masks += (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
