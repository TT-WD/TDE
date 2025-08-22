import os
import pprint
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys

from lib.core.evaluation import decode_preds_tooth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
#from lib.datasets import get_dataset
from lib.core import function
from torchvision.transforms import functional as F

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', default="./experiments/tooth/tooth_hrnet_inference.yaml",
                        help='experiment configuration filename',
                        required=False, type=str)
    parser.add_argument('--model-file', default="./pth_dir/model_best.pth", help='model parameters', required=False,
                        type=str)



    args = parser.parse_args()
    update_config(config, args)
    return args

def init_config():
    #整个应用的初始化配置
    args=parse_args()


def data_transform(img, input_size,):
    """
	该方法用于将图像调整成对应的输入图像的尺寸，同时也要调整对应的关键点的位置
	"""
    # 对图像数据和对应的关键点进行缩放、旋转变换
    ht, wd = img.shape[0], img.shape[1]
    scale = max(ht, wd) * 1.0 / input_size[0]
    # 先对点集和图像进行缩放处理


    new_img = cv2.resize(img, (int(wd * 1.0 / scale), int(ht * 1.0 / scale)))

    # 下面需要填充图像以及，对point_array进行填充偏移了
    padding_x = input_size[0] - int(wd * 1.0 / scale)
    padding_y = input_size[1] - int(ht * 1.0 / scale)
    offset_x = padding_x // 2
    offset_y = padding_y // 2
    img_padding = np.zeros([input_size[0], input_size[1], 3], dtype=np.uint8)
    img_padding[offset_y:offset_y + new_img.shape[0]][offset_x:offset_x + new_img.shape[1]] = new_img


    offset_array = np.array([offset_x, offset_y])
    return img_padding, scale, offset_array


def pre_process_img(img,img_size=512):
    #将img转为对应的tensor
    input,scale,offset_array=data_transform(img,input_size=[img_size,img_size])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input = (input/255.0)
    input =input.transpose((2, 0, 1))
    input_tensor=torch.from_numpy(input).float()
    input_tensor=F.normalize(input_tensor,mean,std)
    return input_tensor,scale,offset_array

def post_process_point_array(point_array,scale,offset_array):
    if isinstance(point_array,np.ndarray):
        if point_array.ndim==2:
            point_array=(point_array-offset_array)*scale
            return point_array
    elif isinstance(point_array,torch.Tensor):
        if point_array.ndim==2:
            point_array=(point_array-offset_array)*scale
            return point_array



def inference_one_img(model,img:np.ndarray):
    #对单张图像进行推理前预处理
    input,scale,offset_array=pre_process_img(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = input.to(device)
    with torch.no_grad():
        input.unsqueeze_(0)
        #图像推理
        output=model(input)
        img_size=torch.tensor(config.MODEL.IMAGE_SIZE).unsqueeze(0)
        heatmap_size=torch.tensor(config.MODEL.HEATMAP_SIZE)
        output_point=decode_preds_tooth(output.data.cpu(),img_size,heatmap_size)
        output_numpy = output_point.squeeze(0).cpu().numpy()
        #推理后处理
        output_numpy=post_process_point_array(output_numpy,scale,offset_array)
        return output_numpy

def get_point_model(model_file="./pth_dir/model_best.pth"):

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    #model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(model_file)
    if isinstance(state_dict, nn.Module):
        model.module = state_dict

    elif 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        try:
            model.module.load_state_dict(state_dict)
        except Exception as e:
            model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    init_config()
    model=get_point_model()
    img=cv2.imread("./data/tooth_new/inference/01guobaohang-0mm.JPG")
    inference_one_img(model,img)