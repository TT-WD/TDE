import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# === Residual Conv Unit ===
class ResidualConvUnit(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1),
		)

	def forward(self, x):
		return x + self.block(x)

# === Chained Residual Pooling ===
class ChainedResidualPool(nn.Module):
	def __init__(self, channels, num_pools=2):
		super().__init__()
		self.blocks = nn.ModuleList()
		for _ in range(num_pools):
			self.blocks.append(
				nn.Sequential(
					nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
					nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
				)
			)

	def forward(self, x):
		out = x
		for block in self.blocks:
			out = out + block(out)
		return out

# === Refine Block（含RCU+CRP+融合）===
class RefineBlock(nn.Module):
	def __init__(self, high_channels, low_channels):
		super().__init__()
		self.rcu_high = ResidualConvUnit(high_channels)
		self.rcu_low = ResidualConvUnit(low_channels)
		self.adapt_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1)
		self.crp = ChainedResidualPool(low_channels)

	def forward(self, higher_feat, lower_feat):
		higher_feat = F.interpolate(self.rcu_high(higher_feat), size=lower_feat.shape[2:], mode='bilinear', align_corners=False)
		fused = self.rcu_low(lower_feat) + self.adapt_conv(higher_feat)
		fused = self.crp(fused)
		return fused

# === RefineNet Decoder ===
class RefineNetDecoder(nn.Module):
	def __init__(self, channels):  # 输入通道：[C2, C3, C4, C5]
		super().__init__()
		self.refine3 = RefineBlock(channels[3], channels[2])
		self.refine2 = RefineBlock(channels[2], channels[1])
		self.refine1 = RefineBlock(channels[1], channels[0])

	def forward(self, feats):  # feats = [C2, C3, C4, C5]
		c2, c3, c4, c5 = feats

		r3 = self.refine3(c5, c4)  # C5+C4
		r2 = self.refine2(r3, c3)  # (C5+C4)+C3
		r1 = self.refine1(r2, c2)  # (C5+C4+C3)+C2

		return [c5,r3,r2,r1]  # 输出多尺度融合特征

# === 整体结构：ResNet34 + RefineNet ===
class ResNet34RefineNet(nn.Module):
	def __init__(self, pretrained=True,config=None):
		super().__init__()
		resnet_type="resnet34"
		if config and config.resnet_type:
			resnet_type=config.resnet_type
		self.encoder = timm.create_model(resnet_type, pretrained=pretrained, features_only=True)
		self.out_channels = [64, 128, 256, 512]  # C2, C3, C4, C5
		self.decoder = RefineNetDecoder(self.out_channels)
		self.decoder.apply(init_weights)
	def forward(self, x):
		feats = self.encoder(x)[1:]  # 取 C2~C5，不要 stem（conv1）
		refined_feats = self.decoder(feats)  # r1, r2, r3

		return feats, refined_feats
def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)