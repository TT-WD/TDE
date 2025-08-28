#基于深度估计模型构建距离估计模型
import json
import math
from typing import Optional

import torch
from torch.nn import AvgPool2d, BatchNorm2d


import torch.nn as nn


from torch.nn import functional as F

from zoedepth.models.toothdis.depth_select_module import PointDepthRefineModule
from zoedepth.models.toothdis.tooth_vit_utils import create_vit, create_pyramid_vit, TransformerDistanceHead, \
	MLPDistanceHead
from zoedepth.models.zoedepth import ZoeDepth
from zoedepth.utils.easydict import EasyDict


class ToothDistanceModel(nn.Module):
	def __init__(self,depth_model,config):


		self.base_config = config
		super(ToothDistanceModel, self).__init__()
		self.depth_model=depth_model

		# 读取 JSON 文件
		with open('./zoedepth/models/toothdis/config_fov_encoder.json', 'r') as f:
			data = json.load(f)
		# 转为 EasyDict（含嵌套支持）
		self.fov_config = to_easydict(data["fov_encoder"])
		self.fov_network=FOVNetwork(self.fov_config,fov_encoder=create_pyramid_vit(self.fov_config,True,))

		self.multi_features_fuse_module=MultiScaleFeatureFusionModule(dims=[256, 256, 256, 256],fuse_type=self.base_config.feature_fuse_type)


		self.depth_select_model=PointDepthRefineModule(32,64,self.base_config.depth_select_patch_size) if self.base_config.depth_select_model else None

		if self.base_config.distance_head_type=="transformer":
			self.distance_head=TransformerDistanceHead(self.base_config)
			self.distance_head.apply(init_weights)
		elif self.base_config.distance_head_type=="mlp":
			self.distance_head=MLPDistanceHead(self.base_config)
		else:
			self.distance_head=None

		self.frozen_module()



	def forward(self,img_data,point_loca):
		'''
			在距离估计网络中，
			1、应该先运行相对估计midas网络，这个是固定不参与训练的，先获取对应的深度估计的编码器、解码器特征，
			2、根据midas的编码器特征去运行FOVNetwork网络，获取焦距，
			3、根据FOVNetwork获得的特征去融合到 到度量深度估计网络中去，确保FOVNetWork 既参加了内参估计，也参与了度量深度结构的微调

			第一版先暂定先把midas的低分辨率编码器特征进行获取，暂时不融合fovnetword的特征到度量学习头中


		Args:
			img_data:
			point_loca:
		Returns:

		'''

		rel_depth,out=self.depth_model.forward_step1(img_data)
		#
		camera_para,camera_features=self.fov_network(img_data,out[1])


		self.multi_features_fuse_module(out[:5],camera_features)


		#用于融合到度量深度估计中的融合深度特征

		result_dict=self.depth_model.forward_step2(img_data,rel_depth,out)
		metric_depth = result_dict["metric_depth"]









		depth_map = metric_depth.squeeze(1) if metric_depth.dim()==4 else metric_depth
		point_loca =point_loca.long()
		# 根据输出的深度图，以及对应的关键点选择出对应点的深度值
		point_loca1 = point_loca[:, :, 0:2]
		point_loca2 = point_loca[:, :, 2:]
		bs = depth_map.shape[0]
		batch_idx = torch.arange(bs,dtype=torch.int64).view(bs, 1).expand(bs, point_loca.shape[1])

		depth_1 = depth_map[batch_idx, point_loca1[:, :, 1], point_loca1[:, :, 0]]*1000   #预测的是m，转换单位为mm
		depth_2 = depth_map[batch_idx, point_loca2[:, :, 1], point_loca2[:, :, 0]]*1000

		# 建立两个3D点的数据
		point_loca1_3d = torch.zeros(size=list(point_loca1.shape)[:-1] + [list(point_loca1.shape)[-1] + 1]).cuda()
		point_loca1_3d[..., -1] = depth_1

		point_loca2_3d = torch.zeros(size=list(point_loca1.shape)[:-1] + [list(point_loca1.shape)[-1] + 1]).cuda()
		point_loca2_3d[..., -1] = depth_2

		# 通过深度回复3D坐标
		# 根据焦距和光心坐标，将2D坐标恢复到3D
		# 恢复point1的Xw和Yw  ,在计算过程中，相机内参是以batch为单位的
		point_loca1_3d[..., 0] = (depth_1 / camera_para[..., 0].unsqueeze(-1).clamp_min(1e-8)) * (
					point_loca1[..., 0] - camera_para[..., 2].unsqueeze(-1))
		point_loca1_3d[..., 1] = (depth_1 / camera_para[..., 1].unsqueeze(-1).clamp_min(1e-8)) * (
					point_loca1[..., 1] - camera_para[..., 3].unsqueeze(-1))

		# 恢复point2的Xw和Yw
		point_loca2_3d[..., 0] = (depth_2 / camera_para[..., 0].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca2[..., 0] - camera_para[..., 2].unsqueeze(-1))
		point_loca2_3d[..., 1] = (depth_2 / camera_para[..., 1].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca2[..., 1] - camera_para[..., 3].unsqueeze(-1))
		origin_distance = torch.norm(point_loca1_3d - point_loca2_3d, dim=-1)
		return origin_distance,metric_depth,rel_depth


	def forward_step1(self,img_data):
		rel_depth, origin_out = self.depth_model.forward_step1(img_data)
		#
		encoder_feats, decoder_feats,crowd_class_out = self.fov_network(img_data, origin_out[1])

		# 用于融合到度量深度估计中的融合深度特征
		#mertric_fused_features = list(reversed(mertric_fused_features))
		fused_out=origin_out
		#融合深度特征以及2D特征
		#fused_out[0:5]=self.multi_features_fuse_module(origin_out[0:5], decoder_feats)


		if self.base_config.fuse_period=="early":
			#用融合后的multi feat生成度量深度
			result_dict = self.depth_model.forward_step2(img_data, rel_depth, fused_out)
		else:
			#用未融合的特征来生成度量深度
			result_dict = self.depth_model.forward_step2(img_data, rel_depth, origin_out)
		metric_depth = result_dict["metric_depth"]
		metric_depth=metric_depth*1000 # 预测的是m，转换单位为mm

		return metric_depth,rel_depth,decoder_feats,fused_out,crowd_class_out
	def forward_step2(self,metric_depth,rel_depth,decoder_feats,point_loca,depth_feats):

		if self.base_config.distance_head_type == "none":
			#没有distance_head，直接用3D公式的逆运算计算
			distance_tuple=self.caculate_dis_by_camera(metric_depth,rel_depth,camera_para,point_loca,feature_map)
			return distance_tuple[0]
		else:
			#通过特有的Transformer head来计算
			# if isinstance(feature_map,list):
			# 	feature_map.extend(rel_depth)
			selected_point_loca=None
			if self.base_config.loss_for_oneTooth:
				selected_point_loca = point_loca[:,:12,...]
			if self.base_config.loss_for_oneTooth and self.base_config.loss_for_archLength:
				selected_point_loca = point_loca[:,:13,...]
			if self.base_config.loss_for_crowd and self.base_config.crowd_loss_type=="regression":
				selected_point_loca = point_loca[:,13:,...]
			if self.base_config.loss_for_crowd and self.base_config.loss_for_archLength and self.base_config.crowd_loss_type=="regression":
				selected_point_loca = point_loca
			if selected_point_loca  is None:
				raise Exception("loss config Error,just three type")

			feature_token=self.point_pair_feature_extractor(depth_feats,metric_depth,decoder_feats,selected_point_loca)
			distance_offset=self.distance_head(feature_token)
			return distance_offset
	def point_pair_feature_extractor(self,depth_feats,metric_depth,decoder_feats,point_match):



		img_h=self.base_config.img_size[0]
		img_w=self.base_config.img_size[1]
		point_match=point_match.float()
		point_match[...,[0,2]]=point_match[...,[0,2]]/img_w
		point_match[...,[1,3]]=point_match[...,[1,3]]/img_h

		features=[*decoder_feats,*depth_feats,metric_depth]
		#features=[*decoder_feats]

		# for idx , encoder_feat in enumerate(encoder_feats):
		# 	features.append(encoder_feat+depth_feats[idx+1])
		# features.extend([depth_feats[-1],depth_feats[0],metric_depth])

		B, N, _ = point_match.shape
		point_match=point_match.view(B,N,2,2)
		feat_list = []


		for feat in features:
			B_, C, H, W = feat.shape
			norm_coords = point_match.clone() * 2 - 1
			norm_coords = norm_coords.view(B, N * 2, 1, 2)

			sampled = F.grid_sample(feat, norm_coords, align_corners=True)
			sampled = sampled.view(B, C, N, 2).permute(0, 2, 3, 1)  # [B, N, 2, C] 2代表两个点，c代表通道数，N代表点对数量
			if self.base_config.point_pair_type == "reduce":
				fused = sampled[:, :, 0, :] + sampled[:, :, 1, :]  # [B, N, C]
			else :
				fused = sampled[:, :, 0, :] + sampled[:, :, 1, :]
			feat_list.append(fused)

		tokens = torch.cat(feat_list, dim=-1)  # [B, N, total_C]
		return tokens



	def caculate_dis_by_camera(self,metric_depth,rel_depth,camera_para,point_loca,feature_map):
		depth_map = metric_depth.squeeze(1) if metric_depth.dim() == 4 else metric_depth
		point_loca = point_loca.long()
		# 根据输出的深度图，以及对应的关键点选择出对应点的深度值
		point_loca1 = point_loca[:, :, 0:2]
		point_loca2 = point_loca[:, :, 2:]
		if self.depth_select_model is not None:
			depth_1 = self.depth_select_model(feature_map, depth_map, point_loca1)
			depth_2 = self.depth_select_model(feature_map, depth_map, point_loca2)
		else:
			bs = depth_map.shape[0]
			batch_idx = torch.arange(bs, dtype=torch.int64).view(bs, 1).expand(bs, point_loca.shape[1])

			depth_1 = depth_map[batch_idx, point_loca1[:, :, 1], point_loca1[:, :, 0]]
			depth_2 = depth_map[batch_idx, point_loca2[:, :, 1], point_loca2[:, :, 0]]

		# 建立两个3D点的数据
		point_loca1_3d = torch.zeros(size=list(point_loca1.shape)[:-1] + [list(point_loca1.shape)[-1] + 1]).cuda()
		point_loca1_3d[..., -1] = depth_1

		point_loca2_3d = torch.zeros(size=list(point_loca1.shape)[:-1] + [list(point_loca1.shape)[-1] + 1]).cuda()
		point_loca2_3d[..., -1] = depth_2

		# 通过深度回复3D坐标
		# 根据焦距和光心坐标，将2D坐标恢复到3D
		# 恢复point1的Xw和Yw  ,在计算过程中，相机内参是以batch为单位的
		point_loca1_3d[..., 0] = (depth_1 / camera_para[..., 0].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca1[..., 0] - camera_para[..., 2].unsqueeze(-1))
		point_loca1_3d[..., 1] = (depth_1 / camera_para[..., 1].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca1[..., 1] - camera_para[..., 3].unsqueeze(-1))

		# 恢复point2的Xw和Yw
		point_loca2_3d[..., 0] = (depth_2 / camera_para[..., 0].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca2[..., 0] - camera_para[..., 2].unsqueeze(-1))
		point_loca2_3d[..., 1] = (depth_2 / camera_para[..., 1].unsqueeze(-1).clamp_min(1e-8)) * (
				point_loca2[..., 1] - camera_para[..., 3].unsqueeze(-1))
		origin_distance = torch.norm(point_loca1_3d - point_loca2_3d, dim=-1)
		return origin_distance, metric_depth, rel_depth

	def get_lr_params(self,lr):
		depth_lr = lr if not self.base_config['depth_lr'] else self.base_config['depth_lr']
		fov_encoder_lr = lr  if not self.base_config['fov_encoder_lr'] else self.base_config['fov_encoder_lr']
		distance_head_lr = lr if not self.base_config['distance_head_lr'] else self.base_config['distance_head_lr']

		dis_model_lr_list=self.depth_model.get_lr_params(depth_lr)

		dis_model_lr_list.append({
			"params":self.fov_network.parameters(),
			"lr":fov_encoder_lr
		})
		if self.distance_head:
			dis_model_lr_list.append({
				"params": self.distance_head.parameters(),
				"lr": distance_head_lr
			})
		return dis_model_lr_list
	# def train(self):
	# 	self.depth_model.train()

	def get_shared_modules_A(self):
		#获取单牙齿宽度、牙弓宽度、拥挤度分类中共享的参数模块
		return self.fov_network.encoder




	def frozen_module(self):
		#根据相关配置判断是否冻结网络，在两阶段训练中会先考虑先进行一次裸训练，第二次在进行随机点价 depth_select
		if not self.base_config.train_depth_model:
			self.depth_model.eval()
			#不训练深度模型，把度量微调模块也冻结了
			for param in self.depth_model.parameters():
				if param.requires_grad:
					param.requires_grad = False

		if not self.base_config.train_fov_network:
			self.fov_network.eval()
			for param in self.fov_network.parameters():
				if param.requires_grad:
					param.requires_grad = False

		if not self.base_config.train_feature_fused_model and self.multi_features_fuse_module :
			self.multi_features_fuse_module.eval()
			for param in self.multi_features_fuse_module.parameters():
				if param.requires_grad:
					param.requires_grad = False

		if not self.base_config.train_depth_select_model and self.depth_select_model:
			self.depth_select_model.eval()
			for param in self.depth_select_model.parameters():
				if param.requires_grad:
					param.requires_grad = False

		if not self.base_config.train_distance_head and self.distance_head :
			if self.distance_head is not None:
				self.distance_head.eval()
				for param in self.distance_head.parameters():
					if param.requires_grad:
						param.requires_grad=False


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

class FOVNetwork(nn.Module):
	"""Field of View estimation network."""

	def __init__(
		self,
		fov_config,
		fov_encoder: nn.Module,
		fov_encoder_type="pyramid",
	):
		"""Initialize the Field of View estimation block.

		Args:
		----
			num_features: Number of features which fov encoder vit use.
			depth_num_features: number of depth features which depth model  midas's encoder out
			fov_encoder: Optional encoder to bring additional network capacity.

		"""
		super().__init__()

		self.fov_config=fov_config
		self.num_features=self.fov_config.fov_num_features
		self.depth_number_feature=self.fov_config.depth_num_features
		self.origin_h=self.fov_config.img_size[0]
		self.origin_w=self.fov_config.img_size[1]
		# Create FOV head.
		#用来对深度特征进行重采样来进行特征融合的,默认深度
		fov_head0 = [
			nn.Conv2d(
				self.depth_number_feature, (self.num_features+self.depth_number_feature)//2, kernel_size=1,
			),  # 384x 8 x 12
			nn.BatchNorm2d((self.num_features+self.depth_number_feature)//2),
			nn.ReLU(True),

			nn.Conv2d(
				(self.num_features+self.depth_number_feature)//2, self.num_features, kernel_size=1,
			),  # 512 x 8 x 12
			nn.BatchNorm2d(self.num_features),
			nn.ReLU(True),

		]
		fov_head = [
			nn.Conv2d(
				self.num_features , self.num_features // 2, kernel_size=3, stride=2, padding=1
			),  # 256 x 4 x 6
			nn.BatchNorm2d(self.num_features // 2),
			nn.ReLU(True),

			nn.Conv2d(
				self.num_features // 2, self.num_features // 4, kernel_size=3, stride=2, padding=1
			),  # 128 x 2 x 3
			nn.BatchNorm2d(self.num_features // 4),
			nn.ReLU(True),
			nn.Conv2d(
				self.num_features // 4, self.num_features // 8, kernel_size=1, stride=1, padding=0
			) if fov_encoder_type == "pyramid" else
			nn.Conv2d(
				self.num_features // 4, self.num_features // 8, kernel_size=3, stride=2, padding=1
			),
			# 64 x 2 x 3
			nn.BatchNorm2d(self.num_features // 8),
			nn.ReLU(True),
			nn.Conv2d(
				self.num_features // 8, 4, kernel_size=(2,3), stride=1, padding=0
			),  # 4 x 1 x 1

		]

		crowd_head= [
			nn.Conv2d(
				self.num_features , self.num_features // 2, kernel_size=3, stride=2, padding=1
			),  # 256 x 4 x 6
			nn.BatchNorm2d(self.num_features // 2),
			nn.ReLU(True),

			nn.Conv2d(
				self.num_features // 2, self.num_features // 4, kernel_size=3, stride=2, padding=1
			),  # 128 x 2 x 3
			nn.BatchNorm2d(self.num_features // 4),
			nn.ReLU(True),
			nn.Conv2d(
				self.num_features // 4, self.num_features // 8, kernel_size=1, stride=1, padding=0
			) if fov_encoder_type == "pyramid" else
			nn.Conv2d(
				self.num_features // 4, self.num_features // 8, kernel_size=3, stride=2, padding=1
			),
			# 64 x 2 x 3
			nn.BatchNorm2d(self.num_features // 8),
			nn.ReLU(True),
			nn.Conv2d(
				self.num_features // 8, 3, kernel_size=(2,3), stride=1, padding=0
			),  # 4 x 1 x 1

		]




		self.encoder = nn.Sequential(
			#embed_dim应该是1024,进行一次特征降维
			fov_encoder, #nn.Linear(fov_encoder.embed_dim, num_features // 2)
		)

		#fov_head0仅仅用来做一次采样，用来融合深度估计编码器特征到 内参估计网络中
		self.resample = nn.Sequential(*fov_head0)

		#fov_head是一个计算相机内参的专用头
		self.camera_head = nn.Sequential(*fov_head)

		self.crowd_head = nn.Sequential(*crowd_head)
		#初始化两个小卷积头
		self.resample.apply(init_weights)

		self.camera_head.apply(init_weights)

		self.crowd_head.apply(init_weights)


		self.freeze_module_by_config()
	def forward(self, x: torch.Tensor, depth_feature: torch.Tensor):
		"""Forward the fov network.
			原来是估计内参的，现在改了，给他用来计算拥挤度类别的

		"""
		#默认只把depth_feature 当作一个常量看待
		if self.fov_config.denorm:
			x = fov_denormalize(x)
			x = fov_normalize(x)
		depth_feature_detach=depth_feature.detach()
		#depth_feature 为midas 的编码器输出的最后一层 channel为256
		#Swin输出的是多尺度层级特征
		camera_feature_list ,fused_feature_list= self.encoder(x)


		#x的特征尺寸为 256*16*24， depth_feature 为1024*16*24
		depth_feature_detach = self.resample(depth_feature_detach)

		head_input_feature = camera_feature_list[-1] + depth_feature_detach

		#先计算相机参数
		camera_para=self.camera_head(head_input_feature)
		camera_para = camera_para.view(camera_para.shape[0], -1)

		# 暂设fx、fy最大为像素的二倍
		camera_para[..., 0] = (torch.tanh(camera_para[..., 0]) + 1) * self.origin_w
		camera_para[..., 1] = (torch.tanh(camera_para[..., 1]) + 1) * self.origin_h
		# cx,cy一般在中心点位置
		camera_para[..., 2] = (self.origin_w // 2) + torch.tanh(camera_para[..., 2]) * (self.origin_w // 10)
		camera_para[..., 3] = (self.origin_h // 2) + torch.tanh(camera_para[..., 3]) * (self.origin_h // 10)

		class_out=self.crowd_head(head_input_feature)
		class_out=class_out.view(class_out.shape[0],-1)


		return camera_para,fused_feature_list,class_out


	def freeze_module_by_config(self):
		if self.fov_config.freeze_config.freeze_fov_encoder:
			#冻结编码器层
			self.encoder.eval()
			for param in self.encoder.parameters():
				param.requires_grad = False
		if self.fov_config.freeze_config.freeze_fov_head:
			self.camera_head.eval()
			for param in self.camera_head.parameters():
				param.requires_grad = False
		if self.fov_config.freeze_config.freeze_crowd_head:
			self.crowd_head.eval()
			for param in self.crowd_head.parameters():
				param.requires_grad= False





class CrossAttentionBlock(nn.Module):
	def __init__(self, dim, num_heads=4):
		super().__init__()
		self.dim = dim
		self.norm_q = nn.LayerNorm(dim)
		self.norm_kv = nn.LayerNorm(dim)
		self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
		self.output_proj = nn.Linear(dim, dim)  # optional projection

	def forward(self, q_feat, kv_feat):
		"""
		q_feat: [B, C, H, W] - from ZoeDepth decoder
		kv_feat: [B, C, H, W] - from PVTv2 encoder
		"""
		B, C, H, W = q_feat.shape
		assert C == self.dim

		# [B, C, H, W] → [B, HW, C]
		q = q_feat.flatten(2).permute(0, 2, 1)   # B, HW, C
		kv = kv_feat.flatten(2).permute(0, 2, 1) # B, HW, C

		# Apply layer norm
		q = self.norm_q(q)
		kv = self.norm_kv(kv)

		# Multihead attention
		attn_out, attn_weights = self.attn(q, kv, kv)  # output: [B, HW, C]

		# Optional projection
		attn_out = self.output_proj(attn_out)

		# [B, HW, C] → [B, C, H, W]
		fused = attn_out.permute(0, 2, 1).view(B, C, H, W)
		return fused



class CrossLinearAttentionBlock(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim
		self.norm_q = nn.LayerNorm(dim)
		self.norm_kv = nn.LayerNorm(dim)
		self.to_q = nn.Linear(dim, dim, bias=False)
		self.to_k = nn.Linear(dim, dim, bias=False)
		self.to_v = nn.Linear(dim, dim, bias=False)
		self.output_proj = nn.Linear(dim, dim)

	def elu_feature_map(self,x):
		# 经典线性注意力中常用的特征映射函数
		return F.elu(x) + 1
	def forward(self, q_feat, kv_feat):
		"""
		q_feat: [B, C, H, W]
		kv_feat: [B, C, H, W]
		"""
		B, C, H, W = q_feat.shape
		assert C == self.dim

		# Flatten and permute to [B, HW, C]
		q = q_feat.flatten(2).permute(0, 2, 1)   # B, N, C where N=H*W
		kv = kv_feat.flatten(2).permute(0, 2, 1) # B, N, C

		# LayerNorm
		q = self.norm_q(q)
		kv = self.norm_kv(kv)

		# Linear projections
		Q = self.to_q(q)  # B, N, C
		K = self.to_k(kv) # B, N, C
		V = self.to_v(kv) # B, N, C

		# Feature map for linear attention
		Q = self.elu_feature_map(Q)  # B, N, C
		K = self.elu_feature_map(K)  # B, N, C

		# Compute KV^T V efficiently:
		# Step1: K^T V  -> (B, C, N) @ (B, N, C) = (B, C, C)
		KV = torch.einsum('bnc,bnd->bcd', K, V)  # b,c,c

		# Step2: Q (KV) -> (B, N, C) @ (B, C, C) = (B, N, C)
		out = torch.einsum('bnc,bcd->bnd', Q, KV)  # B, N, C

		# Normalization factor: Q (K^T 1)
		# Step1: sum over sequence dimension of K
		K_sum = K.sum(dim=1)  # B, C
		# Step2: dot product Q and K_sum
		denom = torch.einsum('bnc,bc->bn', Q, K_sum)  # B, N

		# Avoid division by zero
		denom = denom.unsqueeze(-1).clamp(min=1e-6)  # B, N, 1

		out = out / denom  # normalized output

		# Output projection
		out = self.output_proj(out)  # B, N, C

		# Reshape back to [B, C, H, W]
		out = out.permute(0, 2, 1).view(B, C, H, W)

		return out


class MultiScaleFeatureFusionModule(nn.Module):
	def __init__(self, dims, num_heads=4,fuse_type="attention"):
		super().__init__()
		self.fuse_type = fuse_type
		self.cross_global_blocks = nn.ModuleList([
			CrossAttentionBlock(dim=dim, num_heads=num_heads) for dim in dims[0:2]
		])
		self.cross_linear_blocks = nn.ModuleList([
			CrossLinearAttentionBlock(dim) for dim in dims[2:]   ])
		self.upsample_convs = nn.ModuleList(
			[nn.Conv2d(256, 256, kernel_size=1) for i in range(2)] + [nn.Conv2d(256, 32, kernel_size=1)])
	def forward(self, depth_out, cam_feats):
		if self.fuse_type == "attention":
			return self.forward_with_cross_attention(depth_out, cam_feats)
		elif self.fuse_type == "add":
			return self.forward_with_add(depth_out, cam_feats)

	def forward_with_cross_attention(self, depth_out, cam_feats):
		"""
		depth_out: [1,1/32,1/16,1/8,1/4]
		cam_feats: [1/32,1/16,1/8,1/4]
		returns:
			List of fused features [1,1/32,1/16,1/8,1/4]
		"""


		#根据cam_feats进行query只有
		#分两部分进行特征融合，第一部分低分辨率可以用全局注意力，第二部分高分辨率用linear attention
		depth_out_part1=depth_out[1:3]
		depth_out_part2=depth_out[3:]
		cam_feats_part1=cam_feats[0:2]
		cam_feats_part2=cam_feats[2:]
		fused = []
		for i ,cam_feat in enumerate(cam_feats_part1):
			fused_feat = self.cross_global_blocks[i](depth_out_part1[i], cam_feat)
			fused.append(fused_feat+depth_out_part1[i])
		for idx ,cam_feat in enumerate(cam_feats_part2):
			fused_feat = self.cross_linear_blocks[idx](depth_out_part2[idx], cam_feat)
			fused.append(fused_feat+depth_out_part2[idx])


		fused_feature0 = fused[0]
		for i, upsample_conv in enumerate(self.upsample_convs):
			fused_feature0 = F.interpolate(fused_feature0, scale_factor=2, mode='bilinear')
			fused_feature0 += fused[i + 1]
			fused_feature0 = upsample_conv(fused_feature0)
		fused_feature0 = F.interpolate(fused_feature0, scale_factor=4, mode='bilinear')
		return [fused_feature0+depth_out[0]]+fused
	def forward_with_add(self, depth_out, cam_feats):
		#通过add来实现特征融合
		#depth_out=depth_out.clone()
		#先进行逐级上采样获取与depth[out]相同尺度的特征
		new_depth_out = []
		fused_feature0 = cam_feats[0]
		for i, upsample_conv in enumerate(self.upsample_convs):
			fused_feature0 = F.interpolate(fused_feature0, scale_factor=2, mode='bilinear')
			fused_feature0 += cam_feats[i + 1]
			fused_feature0 = upsample_conv(fused_feature0)

		fused_feature0 = F.interpolate(fused_feature0, scale_factor=4, mode='bilinear')
		depth_out[0] = depth_out[0] + fused_feature0
		depth_out[1] = depth_out[1] + cam_feats[0]
		depth_out[2] = depth_out[2] + cam_feats[1]
		depth_out[3] = depth_out[3] + cam_feats[2]
		depth_out[4] = depth_out[4] + cam_feats[3]

		return depth_out


def to_easydict(d):
	if isinstance(d, dict):
		return EasyDict({k: to_easydict(v) for k, v in d.items()})
	return d






def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)














