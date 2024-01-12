from .st_gcn_aaai18_ordinal_smaller_2_encoder import st_gcn_aaai18_ordinal_smaller_2_encoder
from .st_gcn_POSITION_PRETRAIN_BASE import ST_GCN_18_position_pretrain_model


class ST_GCN_18_ordinal_smaller_2_position_pretrain(ST_GCN_18_position_pretrain_model):
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

    def __init__(self, in_channels, num_class, graph_cfg, **kwargs):
        super().__init__(in_channels, num_class, graph_cfg, st_gcn_aaai18_ordinal_smaller_2_encoder, **kwargs)
