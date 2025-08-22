#对单幅图像进行推理得出距离
import cv2
import numpy as np
import torch

from train_tooth import load_ckpt
from zoedepth.models.builder import build_model
from zoedepth.models.toothdis.tooth_distance_model import ToothDistanceModel
from zoedepth.utils.config import get_config
from torchvision.transforms import functional as F
def img_preprocess(img:np.ndarray,point_loca):
	#进行图像的放缩转换,tianchong
	h, w = img.shape[:2]
	scale = min(256/h, 384/w)
	new_h, new_w = int(h*scale), int(w*scale)
	img=cv2.resize(img, (new_w, new_h))
	pad_x,pad_y = (256-new_h)//2, (384-new_w)//2
	new_img = np.zeros((256, 384, 3), np.uint8)
	new_img[pad_y:new_h+pad_y, pad_x:new_w+pad_x] = img
	new_img=np.asarray(new_img, dtype=np.float32) / 255.0
	input=torch.from_numpy(new_img.transpose((2, 0, 1)))
	mean = [0.5, 0.5, 0.5]
	std = [0.5, 0.5, 0.5]
	F.normalize(input,mean,std)
	input=input.unsqueeze(0)
	#修改对应的关键点
	input_point_loca=(point_loca*scale)+np.array([pad_x,pad_y,pad_x,pad_y]).reshape(1,-1)
	input_point_loca=torch.tensor(input_point_loca,dtype=torch.long)
	input_point_loca=input_point_loca.unsqueeze(0)
	return input,input_point_loca

def get_dis_model():
	config = get_config("toothdis", "infer")
	depth_model = build_model(config)
	distance_model = ToothDistanceModel(depth_model, config)
	config["checkpoint"] = "./tooth_check/only_train_param.pth"
	load_ckpt(config, distance_model)
	return distance_model

def inference_one_img(distance_model,input_img,input_point_loca):
	with torch.no_grad():
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		input_img,input_point_loca=img_preprocess(input_img,input_point_loca)
		distance_model.eval()
		distance_model.to(device)
		input_img=input_img.to(device)
		input_point_loca=input_point_loca.to(device)
		metric_depth, rel_depth, camera_para, out_feat_list, crowd_class_out = distance_model.forward_step1(input_img)
		all_distance_offset_output = distance_model.forward_step2(metric_depth, rel_depth, camera_para,
																  input_point_loca, out_feat_list)
		distance=all_distance_offset_output.cpu().numpy()[0]
		return distance



if __name__ == '__main__':

	config = get_config("toothdis", "infer")
	depth_model = build_model(config)
	distance_model=ToothDistanceModel(depth_model, config)
	config["checkpoint"] = "./tooth_check/only_train_param.pth"
	load_ckpt(config, distance_model)
	origin_point_loca=[[39,79,56,111],[54,114,67,135],[73,136,91,169],[86,178,118,196],
					   [121,197,152,210],[156,210,205,219],[255,220,207,225],[279,186,243,206],
					   [315,365,283,188],[324,124,303,155],[339,103,329,125],[348,62,341,100],
					   ]
	img=cv2.imread("./test_img/03-HJY-U.JPG")
	input_img,input_point_loca=img_preprocess(img,np.array(origin_point_loca))
	with torch.no_grad():
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		distance_model.eval()
		distance_model.to(device)
		input_img=input_img.to(device)
		input_point_loca=input_point_loca.to(device)
		metric_depth, rel_depth, camera_para, out_feat_list, crowd_class_out = distance_model.forward_step1(input_img)
		all_distance_offset_output = distance_model.forward_step2(metric_depth, rel_depth, camera_para,
																  input_point_loca, out_feat_list)
		i=1
