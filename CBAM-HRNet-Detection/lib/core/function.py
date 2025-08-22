# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import logging
from datetime import datetime

import cv2
import torch
import numpy as np

from .evaluation import decode_preds, compute_nme, decode_preds_tooth, compute_nme_tooth

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        loss = critertion(output, target)

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)

        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)

        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    #计算每张图像的nme
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.6f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def train_tooth_debug(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)
        # 得到对应的关键点可见标签


def train_tooth(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict,lr_scheduler):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        #print(f'当前的step为{i}')
        # measure data time
        data_time.update(time.time()-end)
        #得到对应的关键点可见标签
        visible_label= meta["visible"].cuda()
        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        loss = critertion(output, target)

        #只计算可见点的loss
        heatmap_loss=loss*visible_label.unsqueeze(2).unsqueeze(3)
        heatmap_loss=heatmap_loss.mean(dim=(2,3)).sum(dim=(1,))/visible_label.sum(dim=(1,)).clamp(min=1e-6)
        # NME
        heatmap_loss=heatmap_loss.mean()
        score_map = output.data.cpu()
        #preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
        preds = decode_preds_tooth(score_map, meta['input_img_size'], config.MODEL.HEATMAP_SIZE)
        nme_batch = compute_nme_tooth(preds, meta)

        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)

        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        heatmap_loss.backward()
        optimizer.step()

        lr_scheduler.step()
        losses.update(heatmap_loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)

def validate_tooth(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)
            visible_label = meta["visible"].cuda()
            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            heatmap_loss = loss * visible_label.unsqueeze(2).unsqueeze(3)
            heatmap_loss = heatmap_loss.mean(dim=(2, 3)).sum(dim=(1,)) / visible_label.sum(dim=(1,)).clamp(min=1e-6)
            # NME
            heatmap_loss = heatmap_loss.mean()


            preds = decode_preds_tooth(score_map, meta['input_img_size'], config.MODEL.HEATMAP_SIZE)
            # NME
            nme_temp = compute_nme_tooth(preds, meta)
            # Failure Rate under different threshold

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(heatmap_loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions

def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta)


            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions

def inference_tooth(config, data_loader, model,thresholds=None,show_curve=False,auc_targets=[0.5,0.15,0.12, 0.1, 0.08],tooth_num=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()
    point_name_list = ["left-D-1", "left-M-1",
                       "left-D-2", "left-M-2",
                       "left-D-3", "left-M-3",
                       "left-D-4", "left-M-4",
                       "left-D-5", "left-M-5",
                       "left-D-6", "left-M-6",
                       "right-D-1", "right-M-1",
                       "right-D-2", "right-M-2",
                       "right-D-3", "right-M-3",
                       "right-D-4", "right-M-4",
                       "right-D-5", "right-M-5",
                       "right-D-6", "right-M-6",
                       ]
    point_color = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
        (128, 0, 0),  # 深红
        (0, 128, 0),  # 深绿
        (0, 0, 128),  # 深蓝
        (128, 128, 0),  # 橄榄绿
        (128, 0, 128),  # 紫色
        (0, 128, 128),  # 墨绿
        (255, 165, 0),  # 橙色
        (75, 0, 130),  # 靛蓝
        (255, 192, 203),  # 粉色
        (139, 69, 19),  # 棕色
        (127, 255, 0),  # 黄绿色
        (70, 130, 180),  # 钢蓝色
        (218, 112, 214),  # 玫瑰红
        (240, 230, 140)  # 卡其色
    ]
    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    #记录总的归一化后的关键点坐标点的距离误差
    normed_pred_point_total = None
    with torch.no_grad():
        draw_point_dir = os.path.join(config.INFERENCE.OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d&&%H-%M-%S"))
        if not os.path.exists(draw_point_dir):
            os.makedirs(draw_point_dir)


        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds_tooth(score_map, meta['input_img_size'], config.MODEL.HEATMAP_SIZE)



            if config.INFERENCE.DRAW_POINT:
                #如果要在这个上面绘图
                draw_point(preds,draw_point_dir,point_color,point_name_list,meta,config,)
            # NME
            nme_temp = compute_nme_tooth(preds, meta,tooth_num=tooth_num)
            if isinstance(nme_temp, torch.Tensor):
                nme_temp=nme_temp.cpu().numpy()
            if normed_pred_point_total is None:
                normed_pred_point_total = nme_temp
            else:
                normed_pred_point_total =np.concatenate((normed_pred_point_total,nme_temp),axis=0)


            # failure_008 = (nme_temp > 0.08).sum()
            # failure_010 = (nme_temp > 0.10).sum()
            # count_failure_008 += failure_008
            # count_failure_010 += failure_010
            #
            # nme_batch_sum += np.sum(nme_temp)
            # nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if thresholds is None:
        thresholds = np.linspace(0, 0.5, 51)  # 0.00 ~ 0.50

    pck_scores = []
    norm_dists = normed_pred_point_total
    NME=norm_dists.mean()
    for t in thresholds:
        correct = (norm_dists <= t).astype(np.float32)
        pck = correct.mean()
        pck_scores.append(pck)

    pck_scores = np.array(pck_scores)
    auc_results = {}
    for max_thresh in auc_targets:
        idx = np.where(thresholds <= max_thresh)[0]
        if len(idx) < 2:
            print(f"Warning: not enough thresholds below {max_thresh}")
            continue
        sub_thresh = thresholds[idx]
        sub_pck = pck_scores[idx]
        auc_val = np.trapz(sub_pck, sub_thresh) / (sub_thresh[-1] - sub_thresh[0])
        auc_results[f"AUC@{max_thresh:.2f}"] = auc_val

    if show_curve:
        plt.plot(thresholds, pck_scores, label='PCK Curve')
        plt.xlabel("Normalized Distance Threshold")
        plt.ylabel("PCK")
        plt.title("PCK Curve")
        plt.grid(True)
        plt.show()

    return {
        'PCK@0.05': pck_scores[np.searchsorted(thresholds, 0.05)],
        "PCK@0.08 ": pck_scores[np.searchsorted(thresholds, 0.08)],
        'PCK@0.1': pck_scores[np.searchsorted(thresholds, 0.1)],
        **auc_results,
        #'thresholds': thresholds,
        #'pck_scores': pck_scores,
        "NME": NME,
    }

    # nme = nme_batch_sum / nme_count
    # failure_008_rate = count_failure_008 / nme_count
    # failure_010_rate = count_failure_010 / nme_count

    # msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
    #       '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
    #                             failure_008_rate, failure_010_rate)
    # logger.info(msg)

    #return nme, predictions







def draw_point(preds,draw_point_dir,point_color,point_name_list,meta,config,):
    #根据预测的点在对应的图上绘制关键点，
    img_name_list=meta["img_name"]
    N=preds.shape[0] #batch数量
    for i in range(N):
        points= preds[i]
        img=cv2.imread(os.path.join(config.DATASET.ROOT,config.DATASET.INFERENCESET,img_name_list[i]))
        for pn in range(points.shape[0]):

            cv2.circle(img, (int(points[pn][0]), int(points[pn][1])), 3, point_color[pn], -1)
            cv2.putText(img, point_name_list[pn], (int(points[pn][0])+ 50, int(points[pn][1]) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #画完所有的点后保存
        cv2.imwrite(os.path.join(draw_point_dir,img_name_list[i]),img)