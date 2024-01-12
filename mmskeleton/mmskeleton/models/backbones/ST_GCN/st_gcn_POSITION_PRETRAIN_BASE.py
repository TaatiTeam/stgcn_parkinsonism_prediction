import torch.nn as nn

from .st_gcn_aaai18_ordinal_orig_encoder import ST_GCN_18_ordinal_orig_encoder
from .st_gcn_BASE import ST_GCN_model


class ST_GCN_18_position_pretrain_model(ST_GCN_model):
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
                 ST_GCN_encoder,
                 **kwargs):
        super().__init__(in_channels, num_class, graph_cfg, **kwargs)
        print('In ST_GCN_18_ordinal_orig_position_pretrain: ', graph_cfg)
        print(kwargs)
        print("\n")

        self.encoder = ST_GCN_encoder(
            in_channels,
            num_class,
            graph_cfg,
            self.head,
            self.data_bn,
            **kwargs)

        self.stage_2 = False

        # fcn for prediction
        dim_in = self.encoder.output_filters
        dim_in2 = self.encoder.output_filters + self.gait_feat_num
        feat_dim = self.num_joints_predicting * self.in_channels * self.num_ts_predicting

        # the pretrain head predicts each joint location at a future time step
        self.pretrain_head = nn.Conv2d(dim_in, feat_dim, kernel_size=1)

        # The classifcation head is used in stage 2 to predict the clinical score for each walk
        self.classification_head = nn.Conv2d(dim_in2, 1, kernel_size=1)
        self.head = self.pretrain_head
        self.num_class = num_class
