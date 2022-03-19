import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

	def __init__(self, in_channel, out_channel, expand_ratio=4,
		kernel=3, padding=1, stride=1, downsamling=False):
		super().__init__()
		self.downsamling = downsamling

		self.conv1 = nn.Conv2d(
			in_channels=in_channel,
			out_channels=out_channel,
			kernel_size=1,
			stride=1
			)
		self.conv2 = nn.Conv2d(
			in_channels=out_channel,
			out_channels=out_channel,
			kernel_size=kernel,
			padding=padding,
			stride=stride
			)
		self.conv3 = nn.Conv2d(
			in_channels=out_channel,
			out_channels=out_channel*expand_ratio,
			kernel_size=1,
			stride=1
			)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.bn3 = nn.BatchNorm2d(out_channel*expand_ratio)

		self.conv4 = None
		if self.downsamling:
			self.conv4 =nn.Conv2d(
				in_channels=in_channel,
				out_channels=out_channel*expand_ratio,
				kernel_size=1,
				stride=stride
				)


	def forward(self, x):
		
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))

		if self.downsamling:
			x = self.conv4(x)

		out += x
		return out

def residual_blocks(in_channels, out_channels, 
	n_residuals, kernel=3, padding=1, stride=1):

	blocks = []
	for i in range(n_residuals):
		if i == 0:
			blocks.append(Bottleneck(in_channels, out_channels, 
				stride=stride, downsamling=True))
		else:
			blocks.append(Bottleneck(out_channels*4, out_channels))
	return blocks


class ResNet(nn.Module):

	def __init__(self, n_labels):
		super().__init__()
		self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
		self.block2 = nn.Sequential(*residual_blocks(64, 64, 3, stride=2))
		self.block3 = nn.Sequential(*residual_blocks(256, 128, 4, stride=1))
		self.block4 = nn.Sequential(*residual_blocks(512, 256, 6, stride=1))
		self.block5 = nn.Sequential(*residual_blocks(1024, 512, 3, stride=1))
		self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
		self.ffn = nn.Linear(2048, n_labels)


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