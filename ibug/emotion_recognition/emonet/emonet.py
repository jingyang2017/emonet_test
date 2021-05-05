import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class EmoNet(nn.Module):
    def __init__(self, config):
        super(EmoNet, self).__init__()
        self.config = config

        self.emo_convs = []
        self.conv1x1_input_emo_2 = nn.Conv2d(self.config.num_input_channels, 256,
                                             kernel_size=1, stride=1, padding=0)
        for in_f, out_f in [(256, 256)] * self.config.n_blocks:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                      nn.Linear(128, len(self.config.emotion_labels) + self.config.n_reg))

    def forward(self, x):
        emo_feat = self.conv1x1_input_emo_2(x)
        final_features = self.emo_net_2(emo_feat)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        final_features = self.emo_fc_2(final_features)

        return final_features


class AUNET(nn.Module):
    def __init__(self, config):
        super(AUNET, self).__init__()
        self.config = config

        self.emo_convs = []
        self.conv1x1_input_emo_2 = nn.Conv2d(self.config.num_input_channels, 256,
                                             kernel_size=1, stride=1, padding=0)
        for in_f, out_f in [(256, 256)] * self.config.n_blocks:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                      nn.Linear(128, len(self.config.au_labels)))

    def forward(self, x):
        emo_feat = self.conv1x1_input_emo_2(x)
        final_features = self.emo_net_2(emo_feat)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        final_features = self.emo_fc_2(final_features)

        return final_features
