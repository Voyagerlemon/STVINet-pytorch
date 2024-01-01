import os
import torch.nn as nn
import torch.nn.functional as F

from nets.segmentator.deeplabv3.aspp import ASPP
from nets.segmentator.deeplabv3.resnet import ResNet18_OS8


class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()
        self.resnet = ResNet18_OS8()
        self.aspp = ASPP(
            num_classes=self.num_classes)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        output = F.interpolate(output, size=(h, w), mode="bilinear")
        return output

    def create_model_dirs(self):
        self.model_dir = self.project_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/deeplabv3"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)