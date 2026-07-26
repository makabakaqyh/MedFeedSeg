import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


# Unet
class UNet(nn.Module):
    def __init__(self, logits_channel, input_channels=3):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])  # [3,w,h]->[32,w,h]
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])  # [32,w,h]->[64,w,h]
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])  # [64,w,h]->[128,w,h]
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])  # [128,w,h]->[256,w,h]

        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])  # [256,w,h]->[512,w,h]

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])  # [512,w,h]->[256,w,h]
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])  # [256,w,h]->[128,w,h]
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])  # [128,w,h]->[64,w,h]
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])  # [64,w,h]->[32,w,h]

        self.final = nn.Conv2d(nb_filter[0], logits_channel, kernel_size=1)  # [32,w,h]->[1,w,h]

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
