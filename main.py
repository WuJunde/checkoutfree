import numpy as np
from sklearn.manifold import TSNE
import json
import argparse
import matplotlib.pyplot as plt
import math
from bessel import bessel_con
from collections import Counter
from utils import *
from strategy import construct_csg
from nncluster import nnpro
from models.pygcn.train import run_gcn
import cfg
from tqdm import tqdm

args = cfg.parse_args()
per_cont = []
res_labels = {}

class pers():
    def __init__(self, idy, time, fea, loc, ori, label):
        self.times = [time]
        self.cf = add_cf(np.zeros([len(fea)*2 + 1]), fea)
        self.locs = [loc]
        self.oris = [ori]
        self.ids = [idy]
        self.labels = [label]
        self.cid = -1
        self.state = 'cons'

    def add_person(self, per):
        ids, time, cf, loc, ori,label = per.ids, per.times, per.cf, per.locs, per.oris, per.labels
        self.times.extend(time)
        self.cf = add_cf(self.cf, cf[1:257] )
        self.locs.extend(loc)
        self.oris.extend(ori)
        self.ids.extend(ids)
        self.labels.extend(label)

    def set_id(self,cid):
        self.cid = cid
    
    def change_state(self, fea):
        self.oldfea = self.cf[1:257]
        self.cf[1:257] = fea
        self.state = 'disc'

    def change_back(self):
        self.cf[1:257] = self.oldfea
        self.state = 'cons'

    def ident(self, per):
        return sorted(self.ids) == sorted(per.ids)

def nnplay(qurry, per_cont):
    cid, assign, merge_list = nnpro(qurry, per_cont)
    if assign:
        per_cont[cid].add_person(qurry)
    else:
        qurry.set_id(cid)
        per_cont.append(qurry)
    del_list = []
    for (pre_cid, pro_cid) in merge_list:
        per_cont[pro_cid].add_person(per_cont[pre_cid])
        del_list.append(pre_cid)
    for del_cid in del_list:
        del per_cont[del_cid]
    return cid

with open(args.data_path,'r') as infile,open(args.out_path,'r') as outfile :
    snap_dict = json.load(infile)
    count = 0
    with tqdm(total=len(snap_dict), desc=f'progress', unit='piece') as pbar:
        for idy, v in snap_dict.items():
            time, ori, fea, loc, label = v['time'],v['ori'],v['fea'],v['loc'],v['label']
            time, ori, fea, loc = float(time), np.array(json.loads(ori)), np.array(json.loads(fea)), np.array(json.loads(loc))
            qurry = pers(idy, time, fea, loc, ori, label)
            qurry.set_id(count)
            per_cont.append(qurry)
            csg_per, link_map = construct_csg(qurry, per_cont)
            if csg_per == None:
                cid = nnplay(qurry, per_cont)
                res_labels[idy] = cid
                continue
            else:
                newfeas = run_gcn(args, csg_per, link_map)
                newfeas = newfeas.detach().cpu().numpy()
                for i, per in enumerate(csg_per):
                    per.change_state(newfeas[i])
                qurry = csg_per[-1]

                cid = nnplay(qurry, per_cont)
                for i, per in enumerate(csg_per):
                    per.change_back()
                res_labels[idy] = cid
            count += 1
            pbar.update()
    json.dump(res_labels, outfile)
