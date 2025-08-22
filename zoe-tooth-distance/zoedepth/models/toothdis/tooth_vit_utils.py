import timm
import torch
from pyexpat import features
from torch import nn
from torch.nn import functional as F

from zoedepth.models.toothdis.ResRefineNet import ResNet34RefineNet

try:
	from timm.layers import resample_abs_pos_embed
except ImportError as err:
	print("ImportError: {0}".format(err))

def create_vit(
	config,
	use_pretrained: bool = False,
	checkpoint_uri: str =None,
	use_grad_checkpointing: bool = False,
) -> nn.Module:
	"""Create and load a VIT backbone module.

	Args:
	----
		preset: fov_encoder config.
		use_pretrained: Load pretrained weights if True, default is False.
		checkpoint_uri: Checkpoint to load the wights from.
		use_grad_checkpointing: Use grandient checkpointing.

	Returns:
	-------
		A Torch ViT backbone module.

	"""


	'''
		对于fov_encoder
		"dinov2l16_384": ViTConfig(
		in_chans=3,
		embed_dim=1024,
		encoder_feature_layer_ids=[5, 11, 17, 23],
		encoder_feature_dims=[256, 512, 1024, 1024],
		img_size=[256,384],
		patch_size=16,
		timm_preset="vit_large_patch14_dinov2",
		timm_img_size=518,
		timm_patch_size=14,
	),
	'''

	img_size = (config.img_size[0], config.img_size[1])
	patch_size = (config.patch_size, config.patch_size)

	model = timm.create_model(
		config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
	)
	model = make_vit_b16_backbone(
		model,
		encoder_feature_dims=config.encoder_feature_dims,
		encoder_feature_layer_ids=config.encoder_feature_layer_ids,
		vit_features=config.embed_dim,
		use_grad_checkpointing=use_grad_checkpointing,
	)

	if config.patch_size != config.timm_patch_size:
		model.model = resize_patch_embed(model.model, new_patch_size=patch_size)
	if config.img_size != config.timm_img_size:
		model.model = resize_vit(model.model, img_size=img_size)

	if checkpoint_uri is not None:
		state_dict = torch.load(checkpoint_uri, map_location="cpu")
		missing_keys, unexpected_keys = model.load_state_dict(
			state_dict=state_dict, strict=False
		)

		if len(unexpected_keys) != 0:
			raise KeyError(f"Found unexpected keys when loading vit: {unexpected_keys}")
		if len(missing_keys) != 0:
			raise KeyError(f"Keys are missing when loading vit: {missing_keys}")


	return model.model

def create_pyramid_vit(
		config,
		use_pretrained: bool = True,
)-> nn.Module:
	if config.vit_model=='swin':
		return SwinSBackbone(use_pretrained,config)
	elif config.vit_model=='PVTv2_b2':
		return PVTv2B2Backbone(use_pretrained,config)
	elif config.vit_model=='PVTv2_b0':
		return PVTv2B0Backbone(use_pretrained,config)
	elif config.vit_model=='resnet34':
		return ResNet34Backbone(use_pretrained,config)
	elif config.vit_model=='resnet34_refine':
		return ResNet34RefineNet(use_pretrained,config)
class SwinSBackbone(nn.Module):
	def __init__(self, pretrained=True,config=None,):
		super().__init__()
		self.model = timm.create_model(
			'swin_small_patch4_window7_224',
			features_only=False,  # 关键：输出多层特征
			pretrained=pretrained,
			img_size=(config.img_size[0], config.img_size[1]),
			window_size=config.window_size,
		)
		self.stage0 = self.model.patch_embed  # 输出: [B, 96, H/4, W/4]
		self.pos_drop = self.model.pos_drop
		self.layers = self.model.layers  # 4 个 stage
		self.swin_norm = self.model.norm

		self.out_channels = [96, 192, 384, 768]  # Swin-S 的各 stage 输出维度
		self.fuse_norm=nn.ModuleList() #用来给提取的各个层的中间特征加上norm，归一化数据输入
		for dim in self.out_channels:
			self.fuse_norm.append(nn.LayerNorm(dim))
		self.fuse_channels = config.fuse_channels
		self.fused_convs=[ nn.Conv2d(self.out_channels[i],self.fuse_channels[i],1,stride=1,padding=0)  for i in range(len(self.out_channels))]
	def forward(self, x):
		x = self.stage0(x)  # Patch Embedding -> [B, 96, H/4, W/4]
		x = self.pos_drop(x)

		features = []
		for i in range(4):
			x = self.layers[i](x)
			features.append(x.permute(0, 3, 1, 2))  # [B, H*W, C] → [B, C, H, W]

		# List[Tensor]，shape=[B, C_i, H_i, W_i]
		fused_features=[]
		# for i, fused_conv in enumerate(self.fused_convs):
		# 	fused_features.append()
		fused_features=[fused_conv(self.fuse_norm(features[i])) for i, fused_conv in enumerate(self.fused_convs)]
		return features,fused_features

class PVTv2B2Backbone(nn.Module):
	def __init__(self, pretrained=True, config=None,):
		super().__init__()

		# 注意：PVTv2 支持多个变体，可选: pvt_v2_b0, b1, b2, b3, b4
		self.model = timm.create_model(
			'pvt_v2_b2',  # 或者 pvt_v2_b0 以减少计算量
			pretrained=pretrained,
			#features_only=True  # 关键：启用多层输出
		)

		# 每层的输出通道数（取决于模型规模）
		self.out_channels = [64, 128, 320, 512]
		#
		# # 对每层输出进行 LayerNorm 和 Conv1x1 统一维度
		self.before_fuse_norm = nn.ModuleList([
			nn.LayerNorm(c) for c in self.out_channels
		])

		self.fuse_channels = config.fuse_channels
		self.fused_convs = nn.ModuleList([
			nn.Conv2d(c, self.fuse_channels[i], kernel_size=1)
			for i, c in enumerate(self.out_channels)
		])

		#用于多层分辨率构建的最大尺度的融合特征


	def forward(self, x):
		# 获取多层特征：List[Tensor]，每层形状 [B, C_i, H_i, W_i]
		feature_list=[]
		x, feat_size = self.model.patch_embed(x)
		for stage in self.model.stages:
			x, feat_size = stage(x, feat_size=feat_size)
			feature_list.append(x)

		fused_features = []
		for i, feat in enumerate(feature_list):
			B, C, H, W = feat.shape
			feat_ = feat.permute(0, 2, 3, 1)  # → [B, H, W, C]
			normed = self.before_fuse_norm[i](feat_)  # LN
			normed = normed.permute(0, 3, 1, 2)  # → [B, C, H, W]
			fused = self.fused_convs[i](normed)  # Conv1x1 变换通道数到目标通道
			fused_features.append(fused)
		fused_features= list(reversed(fused_features))


		return feature_list,  fused_features


class PVTv2B0Backbone(nn.Module):
	def __init__(self, pretrained=True, config=None,):
		super().__init__()

		# 注意：PVTv2 支持多个变体，可选: pvt_v2_b0, b1, b2, b3, b4
		self.model = timm.create_model(
			'pvt_v2_b0',  # 或者 pvt_v2_b0 以减少计算量
			pretrained=pretrained,
			#features_only=True  # 关键：启用多层输出
		)

		# 每层的输出通道数（取决于模型规模）
		self.out_channels = [32, 64, 160, 256]
		#
		# # 对每层输出进行 LayerNorm 和 Conv1x1 统一维度
		self.before_fuse_norm = nn.ModuleList([
			nn.LayerNorm(c) for c in self.out_channels
		])

		self.fuse_channels = config.fuse_channels
		self.fused_convs = nn.ModuleList([
			nn.Conv2d(c, self.fuse_channels[i], kernel_size=1)
			for i, c in enumerate(self.out_channels)
		])

		#用于多层分辨率构建的最大尺度的融合特征


	def forward(self, x):
		# 获取多层特征：List[Tensor]，每层形状 [B, C_i, H_i, W_i]
		feature_list=[]
		x, feat_size = self.model.patch_embed(x)
		for stage in self.model.stages:
			x, feat_size = stage(x, feat_size=feat_size)
			feature_list.append(x)

		fused_features = []
		for i, feat in enumerate(feature_list):
			B, C, H, W = feat.shape
			feat_ = feat.permute(0, 2, 3, 1)  # → [B, H, W, C]
			normed = self.before_fuse_norm[i](feat_)  # LN
			normed = normed.permute(0, 3, 1, 2)  # → [B, C, H, W]
			fused = self.fused_convs[i](normed)  # Conv1x1 变换通道数到目标通道
			fused_features.append(fused)
		fused_features= list(reversed(fused_features))


		return feature_list,  fused_features

class ResNet34Backbone(nn.Module):
	def __init__(self, pretrained=True, config=None):
		super().__init__()

		# 使用 timm 加载 resnet50
		self.model = timm.create_model(
			'resnet34',
			pretrained=pretrained,
			features_only=True  # 关键：输出中间多层特征
		)
		self.denorm =config.denorm
		# 输出特征的通道数（取决于 resnet 版本）
		self.out_channels = [64, 128, 256, 512]
		self.fuse_norm_list = nn.ModuleList([])

		for dim in self.out_channels:
			self.fuse_norm_list.append(nn.BatchNorm2d(dim))

		# 与 PVTv2 保持一致：对输出通道统一到 config.fuse_channels
		self.fuse_channels = config.fuse_channels


		self.fused_convs = nn.ModuleList([
			nn.Conv2d(c, self.fuse_channels[i], kernel_size=1)
			for i, c in enumerate(self.out_channels)
		])
		# self.cbam_blocks = nn.ModuleList([
		# 	CBAM(self.fuse_channels[i]) for i in range(len(self.out_channels))
		# ])

	# 可选：LayerNorm 不常用于 CNN 输出，通常不加
	# 如果你强行加，也要转为 [B, H, W, C] 形式再加，再 permute 回来

	def forward(self, x):

		# 获取 4 层输出特征 [B, C, H, W]
		feature_list = self.model(x)
		feature_list = feature_list[1:]
		fused_features = []
		for i, feat in enumerate(feature_list):
			# [B, C, H, W] -> Conv1x1 to fuse_channels[i]
			norm_feat=self.fuse_norm_list[i](feat)
			fused = self.fused_convs[i](norm_feat)
			#fused =self.cbam_blocks[i](fused)
			fused_features.append(fused)

		fused_features = list(reversed(fused_features))  # 与 PVTv2 一致，从高层到低层

		return feature_list, fused_features

# CBAM module implementation
class CBAM(nn.Module):
	def __init__(self, channels, reduction=16, kernel_size=7):
		super(CBAM, self).__init__()
		self.channel_att = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, channels // reduction, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(channels // reduction, channels, 1, bias=False),
			nn.Sigmoid()
		)
		self.spatial_att = nn.Sequential(
			nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		# Channel attention
		ca = self.channel_att(x)
		x = x * ca

		# Spatial attention
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		sa = torch.cat([avg_out, max_out], dim=1)
		sa = self.spatial_att(sa)
		x = x * sa
		return x

def make_vit_b16_backbone(
	model,
	encoder_feature_dims,
	encoder_feature_layer_ids,
	vit_features,
	start_index=1,
	use_grad_checkpointing=False,
) -> nn.Module:
	"""Make a ViTb16 backbone for the DPT model."""
	if use_grad_checkpointing:
		model.set_grad_checkpointing()

	vit_model = nn.Module()
	vit_model.hooks = encoder_feature_layer_ids
	vit_model.model = model
	vit_model.features = encoder_feature_dims
	vit_model.vit_features = vit_features
	vit_model.model.start_index = start_index
	vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
	vit_model.model.is_vit = True
	vit_model.model.forward = vit_model.model.forward_features

	return vit_model

def resize_patch_embed(model: nn.Module, new_patch_size=(16, 16)) -> nn.Module:
	"""Resample the ViT patch size to the given one."""
	# interpolate patch embedding
	if hasattr(model, "patch_embed"):
		old_patch_size = model.patch_embed.patch_size

		if (
			new_patch_size[0] != old_patch_size[0]
			or new_patch_size[1] != old_patch_size[1]
		):
			patch_embed_proj = model.patch_embed.proj.weight
			patch_embed_proj_bias = model.patch_embed.proj.bias
			use_bias = True if patch_embed_proj_bias is not None else False
			_, _, h, w = patch_embed_proj.shape

			new_patch_embed_proj = torch.nn.functional.interpolate(
				patch_embed_proj,
				size=[new_patch_size[0], new_patch_size[1]],
				mode="bicubic",
				align_corners=False,
			)
			new_patch_embed_proj = (
				new_patch_embed_proj * (h / new_patch_size[0]) * (w / new_patch_size[1])
			)

			model.patch_embed.proj = nn.Conv2d(
				in_channels=model.patch_embed.proj.in_channels,
				out_channels=model.patch_embed.proj.out_channels,
				kernel_size=new_patch_size,
				stride=new_patch_size,
				bias=use_bias,
			)

			if use_bias:
				model.patch_embed.proj.bias = patch_embed_proj_bias

			model.patch_embed.proj.weight = torch.nn.Parameter(new_patch_embed_proj)

			model.patch_size = new_patch_size
			model.patch_embed.patch_size = new_patch_size
			model.patch_embed.img_size = (
				int(
					model.patch_embed.img_size[0]
					* new_patch_size[0]
					/ old_patch_size[0]
				),
				int(
					model.patch_embed.img_size[1]
					* new_patch_size[1]
					/ old_patch_size[1]
				),
			)

	return model

def resize_vit(model: nn.Module, img_size) -> nn.Module:
	"""Resample the ViT module to the given size."""
	patch_size = model.patch_embed.patch_size
	model.patch_embed.img_size = img_size
	grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
	model.patch_embed.grid_size = grid_size

	pos_embed = resample_abs_pos_embed(
		model.pos_embed,
		grid_size,  # img_size
		num_prefix_tokens=(
			0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
		),
	)
	model.pos_embed = torch.nn.Parameter(pos_embed)

	return model



class TransformerDistanceHead(nn.Module):
	def __init__(self,config ,):
		super().__init__()
		self.config = config
		origin_dim=self.config.origin_dim
		token_dim=self.config.token_dim
		num_tokens=self.config.num_tokens
		hidden_dim=self.config.hidden_dim
		num_layers=self.config.num_layers
		num_heads=self.config.num_heads

		self.projection = nn.Linear(origin_dim, token_dim, bias=False)
		self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_dim))
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=token_dim,
			nhead=num_heads,
			dim_feedforward=hidden_dim,
			batch_first=True
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.output_layer1 = nn.Linear(token_dim, 1)
		self.result_relu = nn.ReLU()

	def forward(self, tokens):
		input_tokens=self.projection(tokens)
		#返回点对直接输出一种偏移误差，在最大最小值之间的
		x = input_tokens + self.pos_embedding
		x = self.transformer(x)
		pred = self.output_layer1(x).squeeze(-1)
		#relative_offset=(torch.tanh(pred)+1)/2
		relative_offset=self.result_relu(pred)
		return relative_offset


class MLPDistanceHead(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config
		self.drop_ratio = self.config.drop_ratio
		self.input_dim = self.config.mlp_dis_head_input_dim
		self.hidden_dim = self.config.mlp_dis_head_hidden_dim
		self.tooth_mlp_head_list = nn.ModuleList()
		if self.config.mlp_head_num>1:
			for i in range(self.config.mlp_head_num):
				self.tooth_mlp_head_list.append(self.build_tooth_mlp_head(out_feat_num=self.config.output_dim))
		else:
			self.tooth_mlp_head_list.append(self.build_tooth_mlp_head(out_feat_num=self.config.output_dim))



	def forward(self, x):
		out=None
		B,N,C=x.shape
		#N代表有N个点对距离
		if N>1 and len(self.tooth_mlp_head_list)>1:
			tooth_out_list=[]
			for idx in range(N):
				tooth_out=self.tooth_mlp_head_list[idx](x[:,idx,:])
				#tooth_out B,4
				tooth_out=tooth_out.unsqueeze(1)
				tooth_out_list.append(tooth_out)
			out= torch.cat(tooth_out_list,dim=1).squeeze(-1)
		else:
			tooth_mlp_head=self.tooth_mlp_head_list[0]
			tooth_out=tooth_mlp_head(x)
			tooth_out = tooth_out.squeeze(-1)
			out=tooth_out
		return out
	def build_tooth_mlp_head(self,out_feat_num=4):
		moduleList=nn.Sequential()
		# mlp1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
		# moduleList.append(mlp1)
		# ln1 = nn.LayerNorm(self.input_dim)
		# moduleList.append(ln1)
		# leaky_relu1 = nn.LeakyReLU(0.1)
		# moduleList.append(leaky_relu1)
		mlp2 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
		moduleList.append(mlp2)
		ln2 = nn.LayerNorm(self.hidden_dim)
		moduleList.append(ln2)
		leaky_relu2 = nn.LeakyReLU(0.1)
		moduleList.append(leaky_relu2)
		dropout2 = nn.Dropout(self.drop_ratio)
		moduleList.append(dropout2)

		mlp3 = nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True)
		moduleList.append(mlp3)
		ln3 = nn.LayerNorm(self.hidden_dim // 2)
		moduleList.append(ln3)
		leaky_relu3 = nn.LeakyReLU(0.1)
		moduleList.append(leaky_relu3)
		dropout2 = nn.Dropout(self.drop_ratio)
		moduleList.append(dropout2)

		mlp4 = nn.Linear(self.hidden_dim//2, out_feat_num, bias=True)
		moduleList.append(mlp4)
		result_relu = nn.Softplus()
		moduleList.append(result_relu)

		return moduleList


def fov_denormalize(x):
	"""Reverses the imagenet normalization applied to the input.

	Args:
		x (torch.Tensor - shape(N,3,H,W)): input tensor

	Returns:
		torch.Tensor - shape(N,3,H,W): Denormalized input
	"""
	mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
	std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
	return x * std + mean

def fov_normalize(x):
	mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
	std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
	return (x - mean)/std