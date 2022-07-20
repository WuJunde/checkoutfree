import numpy as np
from sklearn.manifold import TSNE
import json
import argparse
import matplotlib.pyplot as plt
import math
from bessel import bessel_con
from collections import Counter
import torch

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'vit':
        from vit_pytorch import ViT
        net = ViT(args)
    elif net == 'attention':
        from models.tag.tag_layers import AnyAttention
        net = AnyAttention(4,args.heads)
    elif net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif net == 'unet':
        from models.unet.unet_model import UNet
        net = UNet(args)
    elif net == 'sim':
        from models.sim.simnet import SimNet
        net = SimNet(args)
    elif net == 'efficient':
        from models.efficientnet import EfficientNet
        net = EfficientNet.from_name('efficientnet-b4',gpu_device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net

def cal_es(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def cos_sim(f1,f2):
    t = sum(f1 * f2)
    b = math.sqrt(sum(f1**2)) * math.sqrt(sum(f2**2))
    return t / b

def add_cf(cf,fea):
    cf[0] += 1
    cf[1:len(fea)+1] += fea
    cf[len(fea)+1:] += fea**2
    return cf

def cal_cf(cf):
    l = int((len(cf) -1) / 2)
    m = cf[1:l+1] / cf[0]
    v = math.sqrt(cf[l+1:]/cf[0] - m**2)
    return m,v

def cal_ts(qurry, pr):
    min = 99999
    for i in range(len(pr.ids)):
        t1, t2 = qurry.times[0], pr.times[i]
        z1, z2 = qurry.locs[0], pr.locs[i]
        ez = cal_es(z1,z2)
        res = math.sqrt(ez**2 / 2.8**2 + (t1 - t2)**2)
        if res < min:
            min = res
    return min

def val_vmf(cf_array,fea,case):
    num = 0
    base = 0
    for i in range(len(case)):
        u,k = vmf_es(cf_array[i,:])
        # print("for C: %d, the value of k is: %f" %(i,k))
        fea_i = fea[base:base+case[i]]
        for n in range(case[i]):
            p = vmf_sim(k,u,fea_i[n])
            # print("e %d in c %d, p is %.2f" %(n,i,p))

        num += 1
        base += case[i]
    return

def wlink(qurry, pr):
    qcf,pcf = qurry.cf, pr.cf
    return cal_vmf(qcf,pcf)

def cal_vmf(qcf, pcf):
    u,k = vmf_es(pcf)
    vs = vmf_sim(k,u,qcf)
    return vs

def vmf_es(cf):
    u = cf[1:257] / cf[0]
    r = math.sqrt(np.sum((u**2),axis = 0))

    k = r*(256 - r**2) / (1- r**2)
    return u,k

def vmf_sim(k,u,fea):
    if k == None:
        return cos_sim(u,fea)
    b = bessel_con(k,s = 127)
    tmp = k/(2*math.pi)

    c =  (tmp**128) / (k * b)

    return c * math.exp(k * np.sum(u * fea))

def bcubed(res, img_dict, lab_list):
    p = 0
    r = 0
    for c_id, cluster in enumerate(res):
        gt_lab = []
        for img in cluster['img']:
            gt_lab.append(int(img_dict[img]))
        u = list(set(gt_lab))
        for i in u:
            n = gt_lab.count(i)
            p += n / len(cluster['img'])
            r += n / len(np.where(lab_list == i))
    return p/len(img_dict), r/len(img_dict)


    return

def sne(fea,case):
    fea_embedded = TSNE(n_components=2).fit_transform(fea)
    num = 0
    base = 0
    for i in range(len(case)):

        show_res(base,case[i],fea_embedded,lab_list,num)
        num += 1
        base += case[i]

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(" ")
    plt.legend()
    plt.xlim([-1.25, 1.25])
    plt.ylim([-1.25, 1.25])
    plt.show()
    return

def match(res,img,fea,label):
    """
    para:
    res: list[dict{'cf':,'label':}]
    fea: np.array(256)
    label: the token now
    """
    top_list = []
    max_sim = -1
    max_id = 0
    for id, res_dic in enumerate(res):
        res_cf = res_dic['cf']
        cs = cos_sim(res_cf[1:len(fea)+1] / res_cf[0], fea)
        u,k = vmf_es(res_cf)
        vs = vmf_sim(k,u,fea)
        if len(top_list) < 6:
            top_list.append({'id':id,'sim':cs})
        elif cs > top_list[-1]['sim']:
            top_list[0]['id'] = id
            top_list[0]['sim'] = cs
            top_list = sorted(top_list,key=lambda x:x["sim"])
    # match
    if top_list[-1]['sim'] > 0.85:
        max_id = top_list[-1]['id']
        res[max_id]['cf'] = add_cf(res[max_id]['cf'], fea)
        res[max_id]['img'].append(img)
        label = res[max_id]['label']
        return res, label
    else:
        label = label + 1
        res.append({'img': [img],'cf':add_cf(np.zeros([len(fea)*2 + 1]), fea),'label':label})
        return res, label


def normal(x,y):
    c = np.sqrt(x**2 + y**2)
    return x/c, y/c

def show_res(base,case,fea_embedded,lab_list,num):
    print('base',base)
    print('case',case)
    x1 = fea_embedded[base:base+case,0]
    y1 = fea_embedded[base:base+case,1]
    x1,y1 = normal(x1,y1)
    # plt.plot(x1,y1, color=(col_bar[bar_ind[0]],col_bar[bar_ind[1]],col_bar[bar_ind[2]]), marker='o', label = 'lab' + str(lab))
    plt.plot(x1,y1, color=col_less_bar[num], marker='o', markersize=5)

def gau_sim(m1,v1,m2,v2):
    p = math.sqrt(pai * (v1**2 + v2**2) / 2* (v1**2) * (v2**2))
    b = math.exp(-(m1-m2)**2/(2*(v1**2 + v2**2)))
    return p*b


def lab_2_cf(fea,case):
    num = 0
    base = 0
    cf_array = np.zeros((len(case),256*2 + 1))
    for i in range(len(case)):
        fea_i = fea[base:base+case[i]]
        cf = np.zeros((256*2 + 1))
        for n in range(case[i]):
            cf = add_cf(cf,fea_i[n])
        cf_array[i,:] = cf

        num += 1
        base += case[i]
    return cf_array
