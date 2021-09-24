import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import *
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import ast
import gc
import cv2
MVSDataset = find_dataset_def('data_infer')
test_dataset = MVSDataset(input_format='COLMAP', datapath='/home/hadoop/scx/buaa/test_data/ws_scan9/dense/0', mode="infer", nviews=5, ndepths=512, interval_scale=0.4, max_h=360, max_w=480, both=False, with_colmap_depth_map=False, with_semantic_map=False, have_depth=False, light_idx=3)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, drop_last=False)

test_dataset2 = MVSDataset(input_format='BLEND', datapath='/home/hadoop/scx/mvsnet/mvsnet/test_data/scan9/dense', mode="infer", nviews=5, ndepths=512, interval_scale=0.4, max_h=360, max_w=480, both=False, with_colmap_depth_map=False, with_semantic_map=False, have_depth=False, light_idx=3)
TestImgLoader2 = DataLoader(test_dataset2, 1, shuffle=False, num_workers=0, drop_last=False)

data1 = None
data2 = None
ins, exs = test_dataset.get_camera(0)
print(ins)
# w=480
# h=360
# scale=float(h)/ref_img.shape[0]
# index=int((int(ref_img.shape[1]*scale)-confidence.shape[1])/2)
# flag=0

# if (confidence.shape[1]/ref_img.shape[1]>scale):
#     scale=float(confidence.shape[1])/ref_img.shape[1]
#     index=int((int(ref_img.shape[0]*scale)-confidence.shape[0])/2)
#     flag=1

# #confidence=cv2.pyrUp(confidence)
# ref_img=cv2.resize(ref_img,(int(ref_img.shape[1]*scale),int(ref_img.shape[0]*scale)))
# if (flag==0):
#     ref_img=ref_img[:,index:ref_img.shape[1]-index,:]
# else:
#     ref_img=ref_img[index:ref_img.shape[0]-index,:,:]

# ref_intrinsics, ref_extrinsics = read_camera_parameters('',scale,index,flag)
for batch_idx, sample in enumerate(TestImgLoader):
    print(sample['filename'])
    if sample['filename'][0] == '00000000.jpg':
        data1 = sample['imgs'][0][0]
        #print(sample['proj_matrices'][0][0])
        break

for batch_idx, sample in enumerate(TestImgLoader2):
    print(sample['filename'])
    if sample['filename'][0] == '00000000.jpg':
        data2 = sample['imgs'][0][0]
        #print(sample['proj_matrices'][0][0])
        break

# loss = torch.sum(data1 - data2)
# print(loss)