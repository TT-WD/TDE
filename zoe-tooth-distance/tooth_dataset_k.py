
import json
import re
import traceback
from xml.sax.handler import property_encoding, property_dom_node

import math
import os.path
from pathlib import Path

import cv2
import numpy as np
from torch import tensor
from torch.utils import  data
from torch.utils.data import DataLoader
from torchvision.transforms import RandomVerticalFlip

from tooth_transform_list import *
from torchvision import transforms
#用于验证k折叠的数据集
class ToothDataset(data.Dataset):
	def __init__(self, config, k=5,batch_index_start=0,batch_num=4,is_train=True):

		self.height=config.img_size[0]
		self.width=config.img_size[1]
		# self.min_depth=config.min_depth
		# self.max_depth=config.max_depth
		self.config=config
		self.is_train=is_train
		if config.tooth_flag=="U":
			self.data_dir=config.data_dir_U
			self.one_tooth_min=np.array(config.one_tooth_min_U)
			self.one_tooth_max = np.array(config.one_tooth_max_U)
			self.arch_length_min=np.array(config.arch_length_min_U)
			self.arch_length_max = np.array(config.arch_length_max_U)
		elif config.tooth_flag=="L":
			self.data_dir=config.data_dir_L
			self.one_tooth_min=np.array(config.one_tooth_min_L)
			self.one_tooth_max = np.array(config.one_tooth_max_L)
			self.arch_length_min=np.array(config.arch_length_min_L)
			self.arch_length_max = np.array(config.arch_length_max_L)
		else:
			self.data_dir=config.data_dir
			self.one_tooth_min = np.array(config.one_tooth_min_T)
			self.one_tooth_max = np.array(config.one_tooth_max_T)
			self.arch_length_min = np.array(config.arch_length_min_T)
			self.arch_length_max = np.array(config.arch_length_max_T)
		self.cache_all=config.cache_all
		img_file_list=Path(self.data_dir).glob("*.JPG")
		img_file_name_list=[img_file.name for img_file in img_file_list]

		def extract_number(filename):
			match = re.match(r"(\d+)-", filename)
			return int(match.group(1)) if match else float('inf')

		img_file_name_list=sorted(img_file_name_list,key=extract_number)

		target_img_file_name_list=self.get_kfold_batches(img_file_name_list,k,batch_index_start,batch_num)
		self.img_name_list=[]
		self.proper_json_file_list=[]
		self.existed_json_file_list=[]
		#如果一次缓存所有数据集
		self.img_list=[]
		self.proper_label_list=[]
		self.existed_label_list=[]
		self.depth_array_list=[]
		self.depth_mask_list=[]
		self.transform=ToothTransformer(config)
		self.proper_point_normal_list=["match_0", "match_1", "match_2", "match_3", "match_4", "match_5", "match_6", "match_7", "match_8", "match_9", "match_10", "match_11"]
		self.existed_point_normal_list = ["match_0", "match_1", "match_2", "match_3", "match_4", "match_5", "match_6",]

		self.distance_type= config.distance_type

		for img_file_name in target_img_file_name_list:
			if os.path.exists(os.path.join(self.data_dir,img_file_name.split(".")[0]+".JPG")):
				self.img_name_list.append(img_file_name)

				if self.cache_all:
					self.img_list.append(cv2.imread(os.path.join(self.data_dir, img_file_name)))
					img_h, img_w = self.img_list[-1].shape[:2]
					depth_img_path = os.path.join(self.data_dir, img_file_name.split(".")[0] + "_row_output.png")
					if os.path.exists(depth_img_path):
						self.depth_array_list.append(
							cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32))
					else:
						self.depth_array_list.append(np.ones(shape=(img_h, img_w), dtype=np.float32))
					depth_mask_path = os.path.join(self.data_dir, img_file_name.split(".")[0] + "_mask.jpg")
					if os.path.exists(depth_mask_path):
						self.depth_mask_list.append(cv2.imread(depth_mask_path, cv2.IMREAD_GRAYSCALE))
					else:
						self.depth_mask_list.append(np.zeros(shape=(img_h, img_w), dtype=np.int8))
				#先读取现有单个牙齿的所有标签
				if os.path.exists(os.path.join(self.data_dir,img_file_name.split(".")[0]+".json")):
					#如果对应的img存在对应的json文件，就直接保存对应的json label
					self.proper_json_file_list.append(img_file_name.split(".")[0] + ".json")
					if self.cache_all:
						#如果说一次缓存所有
						self.proper_label_list.append(self.parse_proper_label_from_json(os.path.join(self.data_dir,self.proper_json_file_list[-1])))
				if os.path.exists(os.path.join(self.data_dir,img_file_name.split(".")[0]+"_existed.json")):
					# 如果对应的img存在对应的json文件，就直接保存对应的json label
					self.existed_json_file_list.append(img_file_name.split(".")[0]+"_existed.json")
					if self.cache_all:
						# 如果说一次缓存所有
						try:
							self.existed_label_list.append(self.parse_existed_label_from_json(
								os.path.join(self.data_dir, self.existed_json_file_list[-1])))
						except Exception as e:
							print(f'出现了错误，错误文件为{img_file_name}')
							traceback.print_exc()
							self.existed_label_list.append((1,2,3))
	def __len__(self):
		return len(self.img_name_list)


	def parse_proper_label_from_json(self, json_path):
		match_list=[]
		distance_list=[]
		visible_mask=[]
		#print(f'正在解析label为{json_path}')
		with open(json_path,"r",encoding="utf8") as json_file:
			label_dict=json.load(json_file)

			for match in self.proper_point_normal_list:
				if match in label_dict and "point_loca" in label_dict[match] and (not math.isnan(label_dict[match]["point_distance"]) and label_dict[match]["point_distance"]>0):
					match_list.append(label_dict[match]["point_loca"])

					distance_list.append(label_dict[match]["point_distance"])
					visible_mask.append(1)
				else:
					match_list.append([[0,0],[0,0]])
					distance_list.append(0)
					visible_mask.append(0)
		#读取对应的深度图标签
		distance_array = np.array(distance_list)
		#distance_array=(np.array(distance_list)-self.one_tooth_min)/(self.one_tooth_max-self.one_tooth_min)
		return np.round(np.array(match_list).reshape(-1,4)).astype(np.int32),distance_array,np.array(visible_mask)

	def parse_existed_label_from_json(self, json_path):

		match_list=[]
		distance_list=[]
		visible_mask=[]
		#print(f'正在解析label为{json_path}')
		with open(json_path,"r",encoding="utf8") as json_file:
			label_dict=json.load(json_file)
			if "arch_length" in  label_dict and "match_0" in label_dict and not math.isnan(label_dict["arch_length"]) and label_dict["arch_length"]>0:

				distance_list.append(label_dict["arch_length"])
				visible_mask.append(1)
			else:
				distance_list.append(-1)
				visible_mask.append(0)
			existed_count = sum(1 for i in range(7) if f'match_{i}' in label_dict)
			if existed_count > 5 and not math.isnan(label_dict["sum_distance"]) and label_dict["sum_distance"]>0:
				distance_list.append(label_dict["sum_distance"])
				visible_mask.append(1)
			else:
				distance_list.append(-1)
				visible_mask.append(0)
			for match in self.existed_point_normal_list:
				if match in label_dict :
					match_list.append(label_dict[match]["point_loca"])
				else:
					match_list.append([[0, 0], [0, 0]])

		distance_array=np.array(distance_list)
		#distance_array[0]=(distance_list[0]-self.arch_length_min)/(self.arch_length_max-self.arch_length_min)

		return np.round(np.array(match_list).reshape(-1,4)).astype(np.int32),np.array(distance_list),np.array(visible_mask)

	def __getitem__(self, idx):
		#print(f'label_idx-----{idx}')
		if self.cache_all:
			img=self.img_list[idx]
			img=img.copy()
			depth_array =self.depth_array_list[idx].copy()
			depth_mask = self.depth_mask_list[idx].copy()

		else:
			img=cv2.imread(os.path.join(self.data_dir,self.img_name_list[idx]))
			img_h, img_w = img.shape[:2]
			depth_img_path = os.path.join(self.data_dir, self.img_name_list[idx].split(".")[0] + "_row_output.png")
			if os.path.exists(depth_img_path):
				depth_array=cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
			else:
				depth_array=np.ones(shape=(img_h, img_w), dtype=np.float32)

			depth_mask_path = os.path.join(self.data_dir, self.img_name_list[idx].split(".")[0] + "_mask.jpg")
			if os.path.exists(depth_mask_path):
				depth_mask=cv2.imread(depth_mask_path, cv2.IMREAD_GRAYSCALE)
			else:
				depth_mask=np.zeros(shape=(img_h, img_w), dtype=np.int8)


		img=np.asarray(img, dtype=np.float32) / 255.0

		if self.cache_all:
			proper_point_loca_label, proper_distance_label, proper_visible_mask=self.proper_label_list[idx]
			proper_point_loca_label=proper_point_loca_label.copy()
			proper_distance_label=proper_distance_label.copy()
			proper_visible_mask=proper_visible_mask.copy()


			existed_point_loca_label, existed_distance_label, existed_visible_mask = self.existed_label_list[idx]
			existed_point_loca_label = existed_point_loca_label.copy()
			existed_distance_label = existed_distance_label.copy()
			existed_visible_mask = existed_visible_mask.copy()

		else:
			proper_point_loca_label, proper_distance_label, proper_visible_mask = self.parse_proper_label_from_json(os.path.join(self.data_dir, self.proper_json_file_list[idx]))
			existed_point_loca_label,existed_distance_label,existed_visible_mask=self.parse_existed_label_from_json(os.path.join(self.data_dir,self.existed_json_file_list[idx]))
		proper_data_arrays= self.transform([img,proper_point_loca_label,proper_distance_label,proper_visible_mask,depth_array,depth_mask,existed_point_loca_label],self.is_train)
		proper_data_arrays[1][...,tensor([0,2],dtype=torch.int64)]=proper_data_arrays[1][...,tensor([0,2],dtype=torch.int64)].clamp_(min=0,max=self.width-1)
		proper_data_arrays[1][..., tensor([1, 3], dtype=torch.int64)]=proper_data_arrays[1][..., tensor([1, 3], dtype=torch.int64)].clamp_(min=0, max=self.height - 1)

		existed_point_loca_label=torch.from_numpy(existed_point_loca_label)
		existed_distance_label=torch.from_numpy(existed_distance_label)
		existed_visible_mask=torch.from_numpy(existed_visible_mask)
		existed_point_loca_label[...,tensor([0,2],dtype=torch.int64)]=existed_point_loca_label[...,tensor([0,2],dtype=torch.int64)].clamp_(min=0,max=self.width-1)
		existed_point_loca_label[...,tensor([1, 3], dtype=torch.int64)]=existed_point_loca_label[..., tensor([1, 3], dtype=torch.int64)].clamp_(min=0, max=self.height - 1)


		crowd_label=self.get_crowd_label(proper_distance_label,existed_distance_label)
		crowd_label=torch.from_numpy(crowd_label) if isinstance(crowd_label,np.ndarray) else crowd_label
		#final_mask=(data_arrays[5]>0) & (data_arrays[4]>self.min_depth)  #确保掩膜只关注固定区域内的 深度大于最小深度的区域，避免inf
		#data_arrays[5]=final_mask

		if self.distance_type=='ABS':
			pass
		elif self.distance_type=='REL':
			#如果是相对距离，就进行最大最小归一化
			one_tooth_min=torch.from_numpy(self.one_tooth_min)
			one_tooth_max=torch.from_numpy(self.one_tooth_max)
			arch_min=torch.from_numpy(self.arch_length_min)
			arch_max=torch.from_numpy(self.arch_length_max)
			proper_data_arrays[2]=(proper_data_arrays[2]-one_tooth_min)/(one_tooth_max-one_tooth_min)
			existed_distance_label[0]=(existed_distance_label[0]-arch_min)/(arch_max-arch_min)



		item_dict={
			"img":proper_data_arrays[0],
			"proper_point_loca_label": proper_data_arrays[1],
			"proper_distance_label": proper_data_arrays[2],
			"proper_visible_mask": proper_data_arrays[3],
			"depth_array": proper_data_arrays[4],
			"depth_mask": proper_data_arrays[5],
			"img_name": self.img_name_list[idx],
			"crowd_label": crowd_label,
			"arch_existed_point_loca_label": existed_point_loca_label,
			"arch_existed_distance_label": existed_distance_label,
			"arch_existed_visible_mask": existed_visible_mask,
		}


		return item_dict
	def get_crowd_label(self,proper_distance_label,existed_distance_label):
		proper_sum=proper_distance_label[1:-1].sum()

		if existed_distance_label.shape[0]==2:
			existed_sum=existed_distance_label[1]
		else:
			existed_sum=existed_distance_label
		crowd_value=proper_sum-existed_sum
		if self.config.crowd_label_type == "soft":
			return self.make_soft_label_gaussian(crowd_value)
		else:
			if proper_sum-existed_sum<4:
				return np.array([0])
			elif proper_sum-existed_sum>=4 and proper_sum-existed_sum<8:
				return np.array([1])
			else:
				return np.array([2])

	def make_soft_label_gaussian(self,x, centers=[2, 6, 10], sigma=1.5):
		#x = x.view(-1, 1)
		centers = torch.tensor(centers).to(x.device)
		logits = torch.exp(-((x - centers) ** 2) / (2 * sigma ** 2))
		return logits / logits.sum()

	def get_kfold_batches(self,img_list, k=5, batch_index_start=0, batch_num=4):
		n = len(img_list)
		fold_size = n // k
		remainder = n % k

		# 计算每一折的起止索引
		fold_indices = []
		start = 0
		for i in range(k):
			extra = 1 if i < remainder else 0  # 分配余数
			end = start + fold_size + extra
			fold_indices.append((start, end))
			start = end

		# 收集目标折数数据（循环取模 %k）
		result = []
		for i in range(batch_index_start, batch_index_start + batch_num):
			s, e = fold_indices[i % k]
			result.extend(img_list[s:e])

		return result


class ToothTransformer(object):
	def __init__(self, config,aug_scheduler=None):
		pre_aug_list =[RandomCropNumpy((config.img_size[0], config.img_size[1]))
			]
		self.config = config
		self.aug_scheduler = aug_scheduler
		mid_aug_list=[PointOffsetAug(config)]
		behind_aug_list=[
			RandomHorizontalFlip(),
			[RandomColor(multiplier_range=(0.9, 1.1)), None, None, None, None, None, None],
			ArrayToTensorNumpy(),
			[transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), None, None, None, None, None, None]
		]

		self.pre_train_transform = EnhancedCompose(pre_aug_list)
		self.mid_train_transform = EnhancedCompose(mid_aug_list)
		self.behind_train_transform = EnhancedCompose(behind_aug_list)

		self.test_transform = EnhancedCompose([
			RandomCropNumpy((config.img_size[0], config.img_size[1]),fix_crop=True),
			ArrayToTensorNumpy(),
			[transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), None, None,None,None,None,None]
		])
	def __call__(self, images, train=True):
		if train is True:
			data_arrays=self.pre_train_transform(images)

			if self.aug_scheduler is not None :
				if self.aug_scheduler.start_offset_aug():
					data_arrays=self.mid_train_transform(data_arrays)
			elif self.config.point_offset_aug:
				data_arrays=self.mid_train_transform(data_arrays)

			return self.behind_train_transform(data_arrays)
		else:
			return self.test_transform(images)

#通过k折叠的方式加载数据集
def get_tooth_dataloader_k(config,is_train,k=5,batch_index_start=0,batch_num=4):


	if is_train:
		training_samples = ToothDataset(config, k=k,batch_index_start=batch_index_start,batch_num=batch_num,is_train=is_train)

		if config.distributed:
			train_sampler = torch.utils.data.distributed.DistributedSampler(
				training_samples)
		else:
			train_sampler = None

		data_loader = DataLoader(training_samples,
								 batch_size=config.batch_size,
								 shuffle=(train_sampler is None),
								 num_workers=config.workers,
								 pin_memory=True,
								 #persistent_workers=True,
								 #    prefetch_factor=2,
								 sampler=train_sampler)
	else:
		testing_samples = ToothDataset(config, k=k,batch_index_start=batch_index_start,batch_num=batch_num,is_train=is_train)
		data_loader = DataLoader(testing_samples,
							   config.batch_size, shuffle=False, num_workers=config.workers)
	return data_loader