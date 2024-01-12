import torch.nn as nn

from .st_gcn_ENCODER_BASE import ST_GCN_encoder, st_gcn_block


class st_gcn_aaai18_ordinal_smaller_2_encoder(ST_GCN_encoder):
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

    def __init__(self, in_channels, num_class, graph_cfg, head="stgcn", data_bn=True, **kwargs):
        print("\st_gcn_aaai18_ordinal_smaller_2_encoder")
        super().__init__(in_channels, num_class, graph_cfg, head, data_bn, **kwargs)

        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn_block(in_channels, 32, self.kernel_size, 1, residual=False, **(self.kwargs0)),
                st_gcn_block(32, 32, self.kernel_size, 1, **kwargs),
                st_gcn_block(32, 64, self.kernel_size, 2, **kwargs),
                st_gcn_block(64, 64, self.kernel_size, 1, **kwargs),
            )
        )

        self.output_filters = 64
        self.set_edge_importance()
