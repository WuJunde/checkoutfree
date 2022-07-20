import numpy as np
from sklearn.manifold import TSNE
import json
import argparse
import matplotlib.pyplot as plt
import math
col_bar = [0.05,0.2,0.4,0.6,0.8,0.95]
col_less_bar = ['b','g','r','c','m','y','k',[0.4940, 0.1840, 0.5560]]
col_mask_bar = ['b','w','r','c','m','y','k','g']
col_list = []
for i in range(6):
    for j in range(6):
        for k in range(6):
            col_list.append([i,j,k])

def parse_cmdline():
    p = argparse.ArgumentParser(
        usage="Convert the annotation team's json needed .txt format.")
    p.add_argument('--feature', required=True,
        help='Directory for the original images, required.')
    p.add_argument('--gt', required=True,
        help='Directory for the ground truth images, required.')
    return p.parse_args()

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

def cos_sim(f1,f2):
    t = sum(f1 * f2)
    b = math.sqrt(f1**2) * math.sqrt(f2**2)
    return t / b

def gau_sim(m1,v1,m2,v2):
    p = math.sqrt(pai * (v1**2 + v2**2) / 2* (v1**2) * (v2**2))
    b = math.exp(-(m1-m2)**2/(2*(v1**2 + v2**2)))
    return p*b

def match(res,fea,label):
    top_list = []
    max_sim = -1
    max_id = 0
    for id, res_dic in enumerate(res):
        res_cf = res_dic['cf']
        cos_sim = cos_sim(res_cf[1:len(fea)+1] / res_cf[0], fea)
        if len(top_list) < 6:
            top_list.append({'id':id,'sim':cos_sim})
        elif cos_sim > top_list[-1]['sim']:
            top_list[0]['id'] = max_id
            top_list[0]['sim'] = cos_sim
            top_list = sorted(top_list,key=lambda x:x["sim"])
    # match
    if top_list[-1]['sim'] > 0.6:
        max_id = top_list[-1]['id']
        res[max_id]['cf'] = add_cf(res[max_id]['cf'], fea)
        label = res[max_id]['label']
    else:
        label = label + 1

    # merge
    for ind, mat1 in enmerate(top_list):
        for mat2 in top_list[ind+1:]:
            m1,v1 = cal_cf(res[mat1['id']]['cf'])
            m2,v2 = cal_cf(res[mat2['id']]['cf'])
            cos_sim = cos_sim(m1,m2)


    res.append({'cf':add_cf(np.zeros([len(fea)*2 + 1]), fea),'label':label + 1})
    return res, label + 1


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


def vmf_es(cf):
    u = cf[1:257]
    r = math.sqrt(np.sum(((u / cf[0])**2),axis = 0))
    k = r*(256 - r**2) / (1- r**2)
    return u,k

def main():
    res = []
    cmdline = parse_cmdline()
    label = 0
    length = len(open(cmdline.feature).readlines(  ))
    fea_list = np.zeros((length,256))
    lab_list = np.zeros((length))
    img_dict = {}
    with open(cmdline.gt) as g:
        for line in g:
            img,lab = line.split("\t")
            img_dict[img] = lab
            # print('lab1',lab)
    with open(cmdline.feature) as f:
        num = 0
        for line in f:
            img = list(json.loads(line).keys())[0].split(".")[0]
            lab = img_dict[img]
            # print('lab',lab)
            fea = list(json.loads(line).values())[0]
            fea = np.array(fea)
            fea_list[num,:] = fea
            lab_list[num] = int(lab)-1
            num += 1
            # print('yima processing',num)
    case = []
    for i in [1,3,4,5,6,7,8,10]:
        id_tuple = np.where(lab_list == i-1)
        if i == 1:
            fea = np.squeeze(fea_list[id_tuple,:])
            case.append(len(id_tuple[0]))
        else:
            case.append(len(id_tuple[0]))
            fea = np.concatenate((fea,np.squeeze(fea_list[id_tuple,:])),axis=0)

    cf_array = lab_2_cf(fea,case)
    for i in range(len(cf_array)):
        u,k = vmf_es(cf_array[i,:])
        print("the value of k is",k)

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

if __name__ == '__main__':
    main()
