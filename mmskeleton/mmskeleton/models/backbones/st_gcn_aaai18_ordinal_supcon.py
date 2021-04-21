import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph
from .st_gcn_aaai18_ordinal_orig_encoder import ST_GCN_18_ordinal_orig_encoder

class ST_GCN_18_ordinal_supcon(nn.Module):
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
                 head='stgcn',
                 feat_dim=32,
                 **kwargs):
        super().__init__()
        print('In ST_GCN_18 ordinal supcon: ', graph_cfg)
        print(kwargs)
        self.encoder = ST_GCN_18_ordinal_orig_encoder(
                 in_channels,
                 num_class,
                 graph_cfg,
                 head, 
                 edge_importance_weighting,
                 data_bn,
                 **kwargs)
        self.stage_2 = False
        # fcn for prediction
        dim_in = self.encoder.output_filters
        if head == 'linear':
            self.feature_head = nn.Linear(dim_in, feat_dim)
            self.classification_head =  nn.Linear(dim_in, 1)
        elif head == 'mlp':
            self.feature_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

            self.classification_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, 1)
            )


        elif head =='stgcn':
            self.feature_head = nn.Conv2d(dim_in, feat_dim, kernel_size=1)
            self.classification_head = nn.Conv2d(dim_in, 1, kernel_size=1)

        self.head = self.feature_head
        self.num_class = num_class

    def set_stage_2(self):
        self.head = self.classification_head
        self.stage_2=True

        # print("encoder: ", self.encoder)
        # print('projection head', self.head)
    def forward(self, x):
        # Fine-tuning
        if self.stage_2:
            x = self.encoder(x)
            # prediction
            x = self.head(x)
            x = x.view(x.size(0), -1)
            torch.clamp(x, min=-1, max=self.num_class)


            return x

        # Pretraining
        else:
            feat = self.encoder(x)
            feat = F.normalize(self.head(feat), dim=1)
            return feat