import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


class cnn_custom_2(nn.Module):
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
                 temporal_kernel_size,
                 edge_importance_weighting=True,
                 data_bn=True,
                 input_timesteps=120,
                 **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm3d(1) if data_bn else lambda x: x

        self.temporal_kernel = temporal_kernel_size
        self.conv1_filters = 64
        self.conv2_filters = 128
        self.conv3_filters = 128
        self.conv4_filters = 256
        self.conv5_filters = 256
        self.fc1_out = 256

        dropout_config = {k: v for k, v in kwargs.items() if k == 'dropout'}

        self.dropout = nn.Dropout(p = dropout_config.get('dropout', 0.0))

        # build the CNN
        self.conv1 = nn.Conv3d(1, self.conv1_filters, (1, 13, in_channels))
        self.conv2 = nn.Conv1d(self.conv1_filters, self.conv2_filters, self.temporal_kernel)
        self.conv3 = nn.Conv1d(self.conv2_filters, self.conv3_filters, self.temporal_kernel)
        self.conv4 = nn.Conv1d(self.conv3_filters, self.conv4_filters, self.temporal_kernel)
        self.conv5 = nn.Conv1d(self.conv4_filters, self.conv5_filters, self.temporal_kernel)

        self.num_features_before_fc = (input_timesteps-4*(self.temporal_kernel-1)) * self.conv5_filters
        self.fc1 = nn.Linear(self.num_features_before_fc, self.fc1_out)
        self.output_filters = self.fc1_out



    def forward(self, x):
        # Reshape the input to be of size [bs, 1, timestamps, num_joints, num_coords] 
        x = x.permute(0, 4, 2, 3, 1).contiguous()

        # 3d conv
        x = F.relu(self.conv1(x))
        x = x.squeeze()

        # 1d conv
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))

        x = x.view(-1, self.num_features_before_fc)
        x = F.relu(self.fc1(x))

        return x



