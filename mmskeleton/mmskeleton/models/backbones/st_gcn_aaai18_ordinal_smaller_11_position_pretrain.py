import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph
from .st_gcn_aaai18_ordinal_smaller_11_encoder import ST_GCN_18_ordinal_smaller_11_encoder

class ST_GCN_18_ordinal_smaller_11_position_pretrain(nn.Module):
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
                 temporal_kernel_size = 9,
                 **kwargs):

        super().__init__()
        print('In ST_GCN_18 ordinal supcon: ', graph_cfg)
        print(kwargs)
        self.encoder = ST_GCN_18_ordinal_smaller_11_encoder(
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

        # fcn for prediction
        dim_in = self.encoder.output_filters
        feat_dim = self.num_joints_predicting *2*num_ts_predicting

        # the pretrain head predicts each joint location at a future time step
        self.pretrain_head = nn.Conv2d(dim_in, feat_dim, kernel_size=1)

        # The classifcation head is used in stage 2 to predict the clinical score for each walk
        self.classification_head = nn.Conv2d(dim_in, 1, kernel_size=1)

        self.head = self.pretrain_head
        self.num_class = num_class


    def set_stage_2(self):
        self.head = self.classification_head
        self.stage_2=True

        # print("encoder: ", self.encoder)
        # print('projection head', self.head)
    def forward(self, x):
        if self.in_channels < 3:
            x = x[:, 0:self.in_channels, :, :, :]
        # Fine-tuning
        if self.stage_2:
            x = self.encoder(x)
            # prediction
            x = self.head(x)
            x = x.view(x.size(0), -1)
            torch.clamp(x, min=-1, max=self.num_class)

        # Pretraining
        else:
            # print("============================================")
            # print('input is of size: ', x.size())
            x = self.encoder(x)
            # print('shape of x before encoder is: ', x.size())

            x = self.head(x)
            # print('shape of x before reshaping is: ', x.size())
            # reshape the output to be of size (self.num_joints_predicting x 2 x num_ts)
            x = x.view(x.size(0), 2, self.num_joints_predicting , -1)

            # print('shape of x after reshaping is: ', x.size())

        return x