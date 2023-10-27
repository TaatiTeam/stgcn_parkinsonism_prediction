import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_gcn_ENCODER_BASE import ST_GCN_encoder, st_gcn_block


class ST_GCN_18_ordinal_orig_encoder(ST_GCN_encoder):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 head="stgcn",
                 data_bn=True,
                 **kwargs):
        print("\tST_GCN_18_ordinal_orig_encoder")
        super().__init__(in_channels, num_class, graph_cfg, head, data_bn, **kwargs)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(
                in_channels, 64, self.kernel_size, 1, residual=False, **(self.kwargs0)),
            st_gcn_block(64, 64, self.kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, self.kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, self.kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, self.kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, self.kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, self.kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, self.kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, self.kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, self.kernel_size, 1, **kwargs),
        ))

        self.output_filters = 256

        # initialize parameters for edge importance weighting
        if self.edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        if not self.head == 'stgcn':
            x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

