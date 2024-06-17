import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import mmd
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class MRANNet(nn.Module):

    def __init__(self, in_channels, num_classes=6, bottle_channel=256):
        super(MRANNet, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(48, 64, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(48, 64, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(96, 96, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(96, 96, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 64, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.bottle = nn.Linear(448, bottle_channel)
        self.source_fc = nn.Linear(bottle_channel, num_classes)


    def forward(self, source, target, s_label):

        s_branch1x1 = self.branch1x1(source)  # 32*64*7*7
        s_branch3x3 = self.branch3x3_1(source)
        s_branch3x3 = [
            self.branch3x3_2a(s_branch3x3),
            self.branch3x3_2b(s_branch3x3),
        ]
        s_branch3x3 = torch.cat(s_branch3x3, 1)  # 32*128*7*7
        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = [
            self.branch3x3dbl_3a(s_branch3x3dbl),
            self.branch3x3dbl_3b(s_branch3x3dbl),
        ]
        s_branch3x3dbl = torch.cat(s_branch3x3dbl, 1)  # 32*192*7*7
        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)  # 32*64*7*7

        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch3x3 = self.avg_pool(s_branch3x3)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)

        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch3x3 = s_branch3x3.view(s_branch3x3.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)

        source = torch.cat([s_branch1x1, s_branch3x3, s_branch3x3dbl, s_branch_pool], 1)
        source = self.bottle(source)

        t_branch1x1 = self.branch1x1(target)

        t_branch3x3 = self.branch3x3_1(target)
        t_branch3x3 = [
            self.branch3x3_2a(t_branch3x3),
            self.branch3x3_2b(t_branch3x3),
        ]
        t_branch3x3 = torch.cat(t_branch3x3, 1)

        t_branch3x3dbl = self.branch3x3dbl_1(target)
        t_branch3x3dbl = self.branch3x3dbl_2(t_branch3x3dbl)
        t_branch3x3dbl = [
            self.branch3x3dbl_3a(t_branch3x3dbl),
            self.branch3x3dbl_3b(t_branch3x3dbl),
        ]
        t_branch3x3dbl = torch.cat(t_branch3x3dbl, 1)

        t_branch_pool = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        t_branch_pool = self.branch_pool(t_branch_pool)

        t_branch1x1 = self.avg_pool(t_branch1x1)
        t_branch3x3 = self.avg_pool(t_branch3x3)
        t_branch3x3dbl = self.avg_pool(t_branch3x3dbl)
        t_branch_pool = self.avg_pool(t_branch_pool)

        t_branch1x1 = t_branch1x1.view(t_branch1x1.size(0), -1)
        t_branch3x3 = t_branch3x3.view(t_branch3x3.size(0), -1)
        t_branch3x3dbl = t_branch3x3dbl.view(t_branch3x3dbl.size(0), -1)
        t_branch_pool = t_branch_pool.view(t_branch_pool.size(0), -1)

        target = torch.cat([t_branch1x1, t_branch3x3, t_branch3x3dbl, t_branch_pool], 1)
        target = self.bottle(target)
        # calculate
        output = self.source_fc(source)
        t_label = self.source_fc(target)
        t_label = t_label.data.max(1)[1]
        # (batch, 1)
        cmmd_loss = 0
        if self.training:
            cmmd_loss += mmd.cmmd(s_branch1x1, t_branch1x1, s_label, t_label)
            cmmd_loss += mmd.cmmd(s_branch3x3, t_branch3x3, s_label, t_label)
            cmmd_loss += mmd.cmmd(s_branch3x3dbl, t_branch3x3dbl, s_label, t_label)
            cmmd_loss += mmd.cmmd(s_branch_pool, t_branch_pool, s_label, t_label)
        return output, cmmd_loss
