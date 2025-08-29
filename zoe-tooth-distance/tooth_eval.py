#用于推理、评估牙齿数据的点对距离
import os.path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

#from tooth_dataset import ToothDataset

from tooth_dataset_k import get_tooth_dataloader_k
from train_tooth import load_ckpt
from zoedepth.models.builder import build_model
from zoedepth.models.toothdis.tooth_distance_model import ToothDistanceModel
from zoedepth.utils.config import get_config
import torch
import pandas as pd

from zoedepth.utils.misc import colorize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import ConfusionMatrixDisplay

def compute_crowd_metric_by_regression(proper_distance_output, proper_visible,
									   existed_distance_output, crowd_label,k_index,ul_flag):

	util_index = []
	for i in range(proper_visible.shape[0]):
		if proper_visible[i][1:-1].sum() >= 10:
			util_index.append(i)

	util_proper_output = proper_distance_output[util_index]

	util_existed_output = existed_distance_output[util_index]

	proper_output_sum = util_proper_output[:,1:-1].sum(dim=1)

	existed_output_sum = util_existed_output.sum(dim=1)

	crowd_output = proper_output_sum - existed_output_sum


	crowd_class_output = torch.zeros((crowd_output.shape[0], 3), device=crowd_output.device)

	class1_index = torch.where(crowd_output < 4)[0]
	class2_index = torch.where((crowd_output >= 4) & (crowd_output <= 8))[0]
	class3_index = torch.where((crowd_output >= 8))[0]
	# 构造一个虚假的分类标签
	if len(class1_index) > 0:
		crowd_class_output[class1_index, 0] = 10
	if len(class2_index) > 0:
		crowd_class_output[class2_index, 1] = 10
	if len(class3_index) > 0:
		crowd_class_output[class3_index, 2] = 10
	util_crowd_label = crowd_label[util_index]
	crowd_class_label = util_crowd_label

	preds = torch.argmax(crowd_class_output, dim=1).cpu().numpy()
	labels = crowd_class_label.cpu().numpy()

	# 计算指标
	accuracy = accuracy_score(labels, preds)
	precision = precision_score(labels, preds, average="macro", zero_division=0)
	recall = recall_score(labels, preds, average="macro", zero_division=0)
	f1 = f1_score(labels, preds, average="macro", zero_division=0)
	cm = confusion_matrix(labels, preds, labels=list(range(3)))

	metrics_dict = {
		"accuracy": torch.tensor(accuracy),
		"precision": torch.tensor(precision),
		"recall": torch.tensor(recall),
		"f1": torch.tensor(f1),
		"cm": cm,
	}
	# 绘制混淆矩阵
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(3)))
	disp.plot(cmap="Blues", values_format="d")

	# ✅ 在服务器上保存，不展示
	plt.savefig(f"./confusion_matrix_{ul_flag}_k{k_index}.png", dpi=300, bbox_inches='tight')
	plt.close()  # 关闭绘图，避免资源占用
	return metrics_dict


def decode_one_tooth_rel(config,tooth_dis_output_rel):
	# 对one tooth进行去归一化,变成原始尺寸
	if config.tooth_flag == "U":
		one_tooth_min = np.array(config.one_tooth_min_U)
		one_tooth_max = np.array(config.one_tooth_max_U)
		arch_length_min = np.array(config.arch_length_min_U)
		arch_length_max = np.array(config.arch_length_max_U)
	elif config.tooth_flag == "L":
		data_dir = config.data_dir_L
		one_tooth_min = np.array(config.one_tooth_min_L)
		one_tooth_max = np.array(config.one_tooth_max_L)
		arch_length_min = np.array(config.arch_length_min_L)
		arch_length_max = np.array(config.arch_length_max_L)
	else:
		data_dir = config.data_dir
		one_tooth_min = np.array(config.one_tooth_min_T)
		one_tooth_max = np.array(config.one_tooth_max_T)
		arch_length_min = np.array(config.arch_length_min_T)
		arch_length_max = np.array(config.arch_length_max_T)

	if isinstance(tooth_dis_output_rel, torch.Tensor):
		one_tooth_min = torch.from_numpy(one_tooth_min).to(tooth_dis_output_rel.device)
		one_tooth_max = torch.from_numpy(one_tooth_max).to(tooth_dis_output_rel.device)
		abs_tooth_dis_output = tooth_dis_output_rel * (one_tooth_max - one_tooth_min) + one_tooth_min
		return abs_tooth_dis_output


def decode_arch_length_rel(config, arch_dis_output_rel):
	if isinstance(arch_dis_output_rel, torch.Tensor):
		arch_length_min = torch.from_numpy(config.arch_length_min).to(arch_dis_output_rel.device)
		arch_length_max = torch.from_numpy(config.arch_length_max).to(arch_dis_output_rel.device)
		arch_dis_output = arch_dis_output_rel * (arch_length_max - arch_length_min) + arch_length_min
		return arch_dis_output

if __name__ == "__main__":
	UL_flag="T"
	k_index = 5
	colorize_dir="./colorize_dir"
	config = get_config("toothdis", "infer")
	depth_model = build_model(config)
	distance_model=ToothDistanceModel(depth_model, config)
	config["checkpoint"] = "./tooth_check/only_train_param.pth"
	load_ckpt(config, distance_model)
	#data_test_dir="./dataset/data9"
	# full_state_dict = distance_model.state_dict()
	#
	# # 过滤掉所有属于 depth_model.core 的参数
	# filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('depth_model.core')}
	#
	# # 保存
	# torch.save(filtered_state_dict, 'only_train_param.pth')
	# config.data_dir_val=data_test_dir
	test_loader = get_tooth_dataloader_k(config, False,k=config.k_num,batch_index_start=(k_index+config.batch_num)%config.k_num,batch_num=1)



	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	distance_model.to(device)

	gap=2
	proper_column_names=["image_name", "P-l6", "P-l5", "P-l4", "P-l3", "P-l2", "P-l1", "P-r1", "P-r2", "P-r3", "P-r4", "P-r5", "P-r6", "P-sum", "",
				  "G-l6","G-l5", "G-l4", "G-l3", "G-l2", "G-l1", "G-r1", "G-r2", "G-r3", "G-r4", "G-r5","G-r6","G-sum","","P-G-ABS"
					  ]
	existed_column_name=["image_name","existed_predict","existed_gt","abs"]

	arch_column_name=["image_name","arch_length_predict","arch_length_gt","abs"]

	pre_column_names=None

	distance_row = []

	#计算第几折的索引

	model_type="proper_distance"
	if model_type=="proper_distance":
		excel_path = f"./tooth_eval_proper_{UL_flag}_k{k_index}.xlsx"
		pre_column_names=proper_column_names
	elif model_type=="existed_distance":
		excel_path = f"./tooth_eval_existed_{UL_flag}_k{k_index}.xlsx"
		pre_column_names=existed_column_name
	elif model_type=="arch_length":
		excel_path = f"./tooth_eval_arch_length_{UL_flag}_k{k_index}.xlsx"
		pre_column_names=arch_column_name


	crowd_proper=None
	crowd_existed=None
	# crowd_proper_label=None
	# crowd_existed_label=None
	crowd_proper_visible=None
	crowd_label_sum=None
	for i,batch in enumerate(test_loader):
		with torch.no_grad():
			one_row = []
			images = batch['img'].to(device)

			img_name=batch['img_name'][0]


			proper_point_loca_label = batch['proper_point_loca_label'].to(device)

			proper_distance_label = batch['proper_distance_label'].to(device)

			proper_visible_mask = batch['proper_visible_mask'].to(device)

			arch_existed_point_loca_label = batch['arch_existed_point_loca_label'].to(device)
			arch_existed_distance_label = batch['arch_existed_distance_label'].to(device)
			arch_existed_visible_mask = batch['arch_existed_visible_mask'].to(device)
			crowd_label = batch['crowd_label'].to(device)

			arch_point_local_label, existed_point_local_label = torch.split(arch_existed_point_loca_label,[1, arch_existed_point_loca_label.shape[1] - 1],dim=1)

			arch_distance_label, existed_distance_label = torch.split(arch_existed_distance_label, [1, 1], dim=1)

			arch_length_visible_mask, existed_distance_visible_mask = torch.split(arch_existed_visible_mask, [1, 1],dim=1)

			metric_depth, rel_depth, camera_para, out_feat_list, crowd_class_out = distance_model.forward_step1(images)



			all_point_local = torch.concat([proper_point_loca_label, arch_point_local_label, existed_point_local_label],
										   dim=1)
			all_distance_offset_output = distance_model.forward_step2(metric_depth, rel_depth, camera_para,
																  all_point_local, out_feat_list)

			if config.loss_for_oneTooth or config.loss_for_proper or config.loss_for_crowd:
				proper_distance_offset_output = all_distance_offset_output[:, :12]
				if config.distance_type=="REL":
					proper_distance_offset_output=decode_one_tooth_rel(config,proper_distance_offset_output)
					proper_distance_label = decode_one_tooth_rel(config,proper_distance_label)
			if config.loss_for_archLength:
				arch_length_offset_output = all_distance_offset_output[:, 12]
				if config.distance_type == "REL":
					arch_length_offset_output =decode_arch_length_rel(config,arch_length_offset_output)
			if config.loss_for_existed or config.loss_for_crowd:
				existed_distance_offset_output = all_distance_offset_output[:, 13:]


			if colorize_dir:
				colored = colorize(metric_depth.squeeze().cpu().numpy())
				fpath_colored = os.path.join(colorize_dir,img_name.split(".")[0]+".png")
				Image.fromarray(colored).save(fpath_colored)


			if model_type == "proper_distance":
				one_row.append(img_name)
				#将标签值为0的位置的输出也设置为0
				zero_index=proper_distance_label==0
				proper_distance_offset_output[zero_index]=0

				one_row.extend(list(proper_distance_offset_output.cpu().numpy().squeeze(0)))
				one_row.append(proper_distance_offset_output.sum().item())
				one_row.extend([""] * 1)
				one_row.extend(list(proper_distance_label.cpu().numpy().squeeze(0)))
				one_row.append(proper_distance_label.sum().item())
				one_row.extend([""] * 1)
				one_row.append(abs(proper_distance_label.sum().item() - proper_distance_offset_output.sum().item()))
				distance_row.append(one_row)

			elif model_type == "existed_distance":
				one_row.append(img_name)
				one_row.append(existed_distance_offset_output.sum().view(-1).cpu().item())
				one_row.append(existed_distance_label.view(-1).cpu().item())
				one_row.append(abs(existed_distance_offset_output.sum().view(-1).cpu().item()-existed_distance_label.view(-1).cpu().item()))
				distance_row.append(one_row)
			elif model_type == "arch_length":
				one_row.append(img_name)
				one_row.append(arch_length_offset_output.view(-1).cpu().item())
				one_row.append(arch_distance_label.view(-1).cpu().item())
				one_row.append(abs(arch_length_offset_output.view(-1).cpu().item()-arch_distance_label.view(-1).cpu().item()))
				distance_row.append(one_row)
			elif model_type == "crowd":

				if i==0:
					crowd_proper = proper_distance_offset_output
					crowd_existed = existed_distance_offset_output
					# crowd_existed_label= existed_distance_label
					# crowd_proper_label= proper_distance_label
					crowd_proper_visible=proper_visible_mask
					crowd_label_sum= crowd_label
				else:
					crowd_proper = torch.concat([crowd_proper, proper_distance_offset_output], dim=0)
					crowd_existed = torch.concat([crowd_existed, existed_distance_offset_output], dim=0)
					crowd_label_sum = torch.concat([crowd_label_sum,crowd_label],dim=0)
					crowd_proper_visible=torch.concat([crowd_proper_visible,proper_visible_mask], dim=0)

	if model_type=="crowd":
		metric_dict=compute_crowd_metric_by_regression(crowd_proper,crowd_proper_visible,crowd_existed,crowd_label_sum,k_index,ul_flag=UL_flag)
		print(metric_dict)
		print(f"处理完了第{i}张图像")
	else:
		df = pd.DataFrame(distance_row, columns=pre_column_names)
		df.to_excel(excel_path, index=False)
