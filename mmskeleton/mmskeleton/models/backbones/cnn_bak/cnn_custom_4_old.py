import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


class cnn_custom_4(nn.Module):
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
                 input_timesteps=120,
                 **kwargs):
        super().__init__()
        print('In CNN custom: ', graph_cfg)
        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        self.data_bn = nn.BatchNorm3d(1) if data_bn else lambda x: x


        self.temporal_kernel = 9
        self.conv1_filters = 64
        self.conv2_filters = 128
        self.conv3_filters = 128
        self.fc1_out = 128

        # build the CNN
        self.conv1 = nn.Conv3d(1, self.conv1_filters, (1, 13, 3))
        self.conv2 = nn.Conv1d(self.conv1_filters, self.conv2_filters, self.temporal_kernel)
        self.conv3 = nn.Conv1d(self.conv2_filters, self.conv3_filters, self.temporal_kernel)

        self.num_features_before_fc = (input_timesteps-2*(self.temporal_kernel-1)) * self.conv3_filters

        self.fc = nn.Linear(self.num_features_before_fc, 1)

        self.num_class = num_class
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


    def forward(self, x):
        # Reshape the input to be of size [bs, 1, timestamps, num_joints, num_coords] 
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.data_bn(x)

        # 3d conv
        x = F.relu(self.conv1(x))
        x = x.squeeze()

        # 1d conv
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_features_before_fc)

        x = F.relu(self.fc(x))
        torch.clamp(x, min=-1, max=self.num_class)




        return x



