import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph
from .st_gcn_aaai18_ordinal_orig_encoder import ST_GCN_18_ordinal_orig_encoder


class ST_GCN_18_ordinal_orig_position_pretrain(nn.Module):
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
                 edge_importance_weighting=True,
                 data_bn=True,
                 num_ts_predicting=2,
                 num_joints_predicting=13,
                 head='stgcn',
                 temporal_kernel_size=9,
                 gait_feat_num=0,
                 use_gait_features=True,
                 **kwargs):
        super().__init__()

        self.use_gait_features = use_gait_features
        if not use_gait_features:
            gait_feat_num = 0

        self.encoder = ST_GCN_18_ordinal_orig_encoder(
            in_channels,
            num_class,
            graph_cfg,
            temporal_kernel_size,
            head,
            edge_importance_weighting,
            data_bn,
            **kwargs)
        self.stage_2 = False
        self.num_joints_predicting = num_joints_predicting
        self.in_channels = in_channels
        self.gait_feat_num = gait_feat_num

        # fcn for prediction
        dim_in = self.encoder.output_filters
        dim_in2 = self.encoder.output_filters + gait_feat_num
        feat_dim = self.num_joints_predicting * self.in_channels*num_ts_predicting

        # the pretrain head predicts each joint location at a future time step
        self.pretrain_head = nn.Conv2d(dim_in, feat_dim, kernel_size=1)

        # The classifcation head is used in stage 2 to predict the clinical score for each walk
        self.classification_head = nn.Conv2d(dim_in2, 1, kernel_size=1)
        self.head = self.pretrain_head
        self.num_class = num_class

    def set_classification_head_size(self, num_gait_feats):
        if not self.use_gait_features:
            return
        print('setting classification head to', num_gait_feats)
        self.gait_feat_num = num_gait_feats
        dim_in2 = self.encoder.output_filters + self.gait_feat_num
        self.classification_head = nn.Conv2d(dim_in2, 1, kernel_size=1)

    def set_stage_2(self):
        self.head = self.classification_head
        self.stage_2 = True

    def forward(self, x, gait_feats):
        x = x[:, 0:self.in_channels, :, :, :]


        # Fine-tuning
        if self.stage_2:
            x = self.encoder(x)  # STGCN output

            gait_feats = gait_feats.view(
                gait_feats.size(0), gait_feats.size(1), 1, 1)

            # If we have gait feaures, then combine at the feature level
            if self.use_gait_features:
                x = torch.cat([x, gait_feats], dim=1)

            # prediction
            x = self.head(x)
            x = x.view(x.size(0), -1)

            x[x == float("Inf")] = torch.finfo(x.dtype).max
            x[x == float("-Inf")] = torch.finfo(x.dtype).min
            x[x == float("NaN")] = 0

        # Pretraining
        else:
            x = self.encoder(x)
            x = self.head(x)
            # reshape the output to be of size (13x2xnum_ts)
            x = x.view(x.size(0), self.in_channels,
                       self.num_joints_predicting, -1)


        return x
