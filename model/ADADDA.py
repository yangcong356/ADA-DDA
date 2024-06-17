import torch.nn as nn
from .backbone.utils import mmd
from .backbone import resnet_triplet_att as resnet_att
from .backbone.MRAN import MRANNet

class ADADDA(nn.Module):
    def __init__(self, args):
        super(ADADDA, self).__init__()
        self.sharedNet = resnet_att.__dict__[args.backbone](pretrained=False, num_classes=args.num_classes, att_type=args.att_type)
        self.MRANNet = MRANNet(2048, args.num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, source, target, s_label):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        
        source_pred, cmmd_loss = self.MRANNet(source, target, s_label)
        mmd_loss = 0
        if self.training:
            source = self.avgpool(source)
            source = source.view(source.size(0), -1)
            target = self.avgpool(target)
            target = target.view(target.size(0), -1)
            mmd_loss += mmd.mmd_rbf_noaccelerate(source, target)
        return source_pred, cmmd_loss, mmd_loss