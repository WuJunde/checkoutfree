import numpy as np
import cv2 as cv
import argparse
import os.path
import pathlib
import logging
import math
import json
import cfg
from models.tag.tag import Stage
import torch
from torch import nn
from utils import *

args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)

def construct_csg(qurry, per_cont):
    picked_per = pick(qurry, per_cont)
    if picked_per:
        link_map = link(qurry, picked_per)
        csg_per, link_map = weight(qurry, picked_per, link_map)
        return csg_per, link_map
    else:
        return None, None

def dic_link(oris1, locs1, oris2, locs2):
    oris1, locs1, oris2, locs2 = torch.FloatTensor(oris1).to(device = GPUdevice),torch.FloatTensor(locs1).to(device = GPUdevice),torch.FloatTensor(oris1).to(device = GPUdevice),torch.FloatTensor(locs2).to(device = GPUdevice)
    net = get_network(args, 'attention', gpu_device = GPUdevice)
    rpn_qpos, rpn_kpos = nn.Parameter(torch.Tensor(1, 1, 1, 4)), nn.Parameter(torch.Tensor(1, 1, 1, 4))
    rpn_qpos = rpn_qpos.expand(1, -1, -1, -1)
    rpn_kpos = rpn_kpos.expand(1, -1, -1, -1)
    attn_out = 0
    for i in range(len(oris1)):
        o1p, l1p, o2p, l2p = oris1[i], locs1[i], oris2[i], locs2[i]
        attn_out += net(q=torch.cat((o1p, l1p),-1).view(1, 1, 4), k=torch.cat((o2p, l2p),-1).view(1,  1, 4), v=torch.cat((o2p, l2p),-1).view(1, 1, 4), qpos=rpn_qpos, kpos=rpn_kpos)
    attn_out = attn_out / len(oris1)
    attn_out = nn.Linear(4, 1).to(device=GPUdevice)(attn_out)
    return nn.Sigmoid()(torch.squeeze(attn_out))

def pick(qurry, per_cont):
    picked_per = []
    for ind, pr in reversed(list(enumerate(per_cont))):
        if len(picked_per) > args.pickwin-2:
            break
        ts = cal_ts(qurry, pr)
        # print('ts is',ts)
        if ts < args.tsth:
            picked_per.append(pr)
    if len(picked_per) < args.pickwin-1: #incomplete graph
        return
    return picked_per

def link(qurry, csg_per):
    csg_per.append(qurry)
    link_map = np.zeros((len(csg_per),len(csg_per)))
    for i, pr1 in enumerate(csg_per):
        for j, pr2 in enumerate(csg_per):
            if i == j:
                link_map[i][j] = 1
                continue
            linked = dic_link(pr1.oris, pr1.locs, pr2.oris, pr2.locs)
            link_map[i][j] = linked

    return link_map

def weight(qurry, csg_per, link_map):
    ind_tuple = np.where(link_map == 1)
    for ind in range(len(ind_tuple[0])):
        x_ind = ind_tuple[0][ind]
        y_ind = ind_tuple[1][ind]
        if x_ind == y_ind:
            link_map[x_ind][y_ind] = 1
            continue
        pera = csg_per[x_ind]
        perb = csg_per[y_ind]

        w = wlink(pera, perb)

        link_map[x_ind][y_ind] = w

    return csg_per, link_map
