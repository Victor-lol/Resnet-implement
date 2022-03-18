import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel=3, padding=1,use_1conv=False,strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel, 
            kernel_size=kernel, 
            padding=padding, 
            stride=strides)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=kernel, 
            padding=padding)
    
        if use_1conv:
            self.conv3 = nn.Conv2d(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=1,
                stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        out += x
        return out

def resnet_block(in_channels, out_channels, n_residuals, kernel=3, padding=1, first_block=False):
    blocks = []
    for i in range(n_residuals):
        if i == 0 and not first_block:
            blocks.append(Residual(in_channels, out_channels,
                                use_1conv=True, strides=2))
        else:
            blocks.append(Residual(out_channels, out_channels))
    return blocks
    

class ResNet(nn.Module):
    
    def __init__(self, n_labels):
        super().__init__()
        self.n_labels = n_labels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(*resnet_block(64,64,3,first_block=True))
        self.block3 = nn.Sequential(*resnet_block(64,128,4))
        self.block4 = nn.Sequential(*resnet_block(128,256,6))
        self.block5 = nn.Sequential(*resnet_block(256,512,3))
        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.ffn = nn.Linear(512, n_labels)

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgPool(x)
        x = nn.Flatten()(x)
        x = self.ffn(x)

        return x
