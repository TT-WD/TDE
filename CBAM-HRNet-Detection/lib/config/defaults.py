# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0, 1, 2, 4)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 20
_C.MODEL.TARGET_TYPE = 'Gaussian'
_C.MODEL.IMAGE_SIZE = [512, 512]  # width * height
_C.MODEL.HEATMAP_SIZE = [256, 256]  # width * height
_C.MODEL.SIGMA = 1.5
_C.MODEL.EXTRA = CN()
_C.MODEL.START_CBAM = True

# High-Resoluion Net
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.EXTRA.STEM_INPLANES = 64
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.WITH_HEAD = True

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 32, 72, 144]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'



# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = './data'
_C.DATASET.DATASET = '300w'
_C.DATASET.TRAINSET = 'tooth/train'
_C.DATASET.VALSET = 'tooth/test'
_C.DATASET.INFERENCESET= 'tooth/inference'
# training data augmentation
_C.DATASET.FLIP = False
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001
_C.TRAIN.LR_TYPE="CosineAnnealingLR"
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.01
#对于warmup余弦退火调度器的相关配置
_C.TRAIN.PCT_START=0.05
_C.TRAIN.MIN_LR_RATIO= 0.0001
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.SAVE_EPOCH = 20
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 20
_C.TEST.DRAW_POINT = False
_C.INFERENCE=CN()

_C.INFERENCE.OUTPUT_DIR = ''
_C.INFERENCE.DRAW_POINT= False

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
