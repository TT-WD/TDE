#这是一个增强调度器，用来控制何时进行点偏移增强以及是否使用depth_select_module

import math
class AugScheduler:
	def __init__(self,config):
		self.config=config
		self.scheduler_type=config.scheduler_type
		self.offset_aug_start_epoch=config.offset_aug_start_epoch
		self.depth_select_start_epoch=config.depth_select_start_epoch
		self.offset_start_mae=config.offset_start_mae
		self.depth_select_start_mae=config.depth_select_start_mae
		self.pre_epoch=0
		self.pre_mae=math.inf
	def start_offset_aug(self):
		if self.scheduler_type=='epoch' and self.pre_epoch>self.offset_aug_start_epoch:
			return True
		elif self.scheduler_type=='mae' and self.pre_mae<self.offset_start_mae:
			return True
		else:
			return False
	def start_depth_select_aug(self):
		if self.scheduler_type=='epoch' and self.pre_epoch>self.depth_select_start_epoch:
			return True
		elif self.scheduler_type=='mae' and self.pre_mae<self.depth_select_start_mae:
			return True
		else:
			return False
	def set_pre_epoch(self,epoch):
		self.pre_epoch=epoch
	def set_pre_mae(self,mae):
		self.pre_mae=mae