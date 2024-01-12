import torch
import torch.nn as nn

from icecream import ic
class ST_GCN_model(nn.Module):
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
        self.__set_default_params(**kwargs)

        self.in_channels = in_channels
        self.num_class = num_class
        self.graph_cfg = graph_cfg

        if not self.use_gait_features:
            self.gait_feat_num = 0

        super().__init__()

    def __set_default_params(self, **kwargs):
        # Set to default values unless provided

        default_dict = {
            "edge_importance_weighting": True,
            "data_bn": True,
            "num_ts_predicting": 2,
            "num_joints_predicting": 13,
            "head": "stgcn",
            "temporal_kernel_size": 9,
            "gait_feat_num": 0,
            "use_gait_features": False,
        }

        # Update if we have value
        for k, v in default_dict.items():
            if k in kwargs:
                self.__dict__[k] = kwargs[k]
            else:
                self.__dict__[k] = v

    def set_classification_head_size(self, num_gait_feats):
        if not self.use_gait_features:
            return
        print("setting classification head to", num_gait_feats)
        self.gait_feat_num = num_gait_feats
        dim_in2 = self.encoder.output_filters + self.gait_feat_num
        self.classification_head = nn.Conv2d(dim_in2, 1, kernel_size=1)

    def set_stage_2(self):
        self.head = self.classification_head
        self.stage_2 = True

    def forward(self, x, gait_feats):
        # ic(x)
        ic(x.shape)
        x = x[:, 0 : self.in_channels, :, :, :]

        ic(x.shape)
        quit()
        # Fine-tuning
        if self.stage_2:
            x = self.encoder(x)  # STGCN output

            gait_feats = gait_feats.view(gait_feats.size(0), gait_feats.size(1), 1, 1)

            # If we have gait feaures, then combine at the feature level
            if self.use_gait_features:
                x = torch.cat([x, gait_feats], dim=1)

            # prediction
            x = self.head(x)
            x = x.view(x.size(0), -1)

            x[x == float("Inf")] = torch.finfo(x.dtype).max
            x[x == float("-Inf")] = torch.finfo(x.dtype).min
            x[x == float("NaN")] = 0

            # x = torch.clamp(x, min=-1, max=self.num_class)

        # Pretraining
        else:
            # print("============================================")
            # print('input is of size: ', x.size())
            x = self.encoder(x)

            x = self.head(x)
            # print('shape of x before reshaping is: ', x.size())
            # reshape the output to be of size (13x2xnum_ts)
            x = x.view(x.size(0), self.in_channels, self.num_joints_predicting, -1)

            # print('shape of x after reshaping is: ', x.size())

        return x

    pass
