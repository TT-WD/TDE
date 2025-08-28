# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np
from torch import Tensor

from ..utils.transforms import transform_preds, transform_preds_tooth


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    #获取model推理得到的热力图的坐标
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds
def compute_nme_tooth(preds, meta,tooth_num=1):
    targets = meta['input_point_array']
    visible_array = meta["visible"]
    preds = preds.numpy()
    target = targets.cpu().numpy()
    visible_array=visible_array.cpu().numpy()
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    interoculars=np.zeros(N)
    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 24:  # tooth数据集
            interocular_1 = np.linalg.norm(pts_gt[0,] - pts_gt[1,])
            interocular_2 = np.linalg.norm(pts_gt[12,] - pts_gt[13,])
            if tooth_num == 1:
                interocular= (interocular_1 + interocular_2) / 2

            else:
                if interocular_1 <5 or interocular_2 <5:
                    interocular=max(interocular_1, interocular_2)*2
                else:
                    interocular=interocular_1+interocular_2
            interoculars[i] = interocular
        else:
            raise ValueError('Number of landmarks is wrong')
    interoculars[interoculars==0] = interoculars[interoculars!=0].mean()
        # 只计算可见关键点的损失和mse
    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1) * visible_array[i]) / (
                    interoculars[i] * np.sum(visible_array[i]))
    return rmse

def compute_nme(preds, meta):
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def decode_preds_tooth(score_map, input_size_batch, heatmap_res):
    #该函数是为了解码热力图的坐标，映射到原始图像坐标，为了后续计算NME
    input_size_batch=input_size_batch.numpy() if isinstance(input_size_batch, Tensor) else input_size_batch
    coords = get_preds(score_map)
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = score_map[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < heatmap_res[0]) and (py > 1) and (py < heatmap_res[1]):
                # 对对应的关键点坐标进行修正,px,py一定是大于1的
                # 热力图的像素坐标是匆匆1,1开始的不是从0,0
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    #需要将一个batch的坐标预测值，转换到原始图中的坐标大小
    for i in range(coords.size(0)):
        preds[i] = transform_preds_tooth(coords[i], input_size_batch[i], heatmap_res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


def decode_preds(output, center, scale, res):
    #将一个batch的输出的热力图转换为对应的最大概率可能的对应坐标
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                #对对应的关键点坐标进行某种修正,px,py一定是大于1的
                #热力图的像素坐标是匆匆1,1开始的不是从0,0
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()


    # Transform back
    for i in range(coords.size(0)):
        #coord[i]代表一张图片的输出热力图
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
