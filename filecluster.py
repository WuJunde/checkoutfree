import json
from scipy import spatial
import math
import numpy as np

scr_fp = '...'
out_path = '...'
rep = []
thnna = 0.65
thnnm = 0.86



def cos_sim(f1,f2):
    t = sum(f1 * f2)
    b = math.sqrt(sum(f1**2)) * math.sqrt(sum(f2**2))
    return t / b

def gau_sim(m1,v1,m2,v2):
    p = math.sqrt(pai * (v1**2 + v2**2) / 2* (v1**2) * (v2**2))
    b = math.exp(-(m1-m2)**2/(2*(v1**2 + v2**2)))
    return p*b


def add_cf(cf,fea):
    cf[0] += 1
    cf[1:len(fea)+1] += fea
    cf[len(fea)+1:] += fea**2
    return cf

def cal_cf(cf):
    l = int((len(cf) -1) / 2)
    m = cf[1:l+1] / cf[0]
    # print('cf[l+1:]',cf[l+1:]/cf[0]- m**2)
    # print('cf[0]',cf[0])
    # print('m',m**2)
    v = np.sqrt(cf[l+1:]/cf[0] - m**2)
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


def nna(fea): #return the cluster id of the qurry
    if len(rep) == 0:
        rep.append(add_cf(np.zeros([len(fea)*2 + 1]), fea))
        return 0
    for ind, pr in reversed(list(enumerate(rep))):
        m,v = cal_cf(pr)
        score = 1 - spatial.distance.cosine(fea, m)
        if score > thnna: #match and absorb
            rep[ind] = add_cf(pr,fea)
            return ind, 'assign'
    rep.append(add_cf(np.zeros([len(fea)*2 + 1]), fea)) #no find, new person
    return len(rep) - 1, 'new'

def nnm(rep):
    mudict = {}
    numlist = []
    for ind, pr in reversed(list(enumerate(rep))):
        m,v = cal_cf(pr)
        for i in range(ind):
            fea,v2 = cal_cf(rep[i])
        score = 1 - spatial.distance.cosine(fea, m)
        if score > thnnm: #match and absorb
            newph = add_cf(pr,fea)
            rep[ind] = newph
            rep[i] = newph

def nnpro(qurry, per_cont):
    fea = qurry.cf
    fea = np.array(fea)
    cid,act = nna(fea)
    if act
    nnm(rep)
    return cid

def file_inter(scr_fp, out_path):
    with open(scr_fp, 'r') as infile, open(out_path, 'w') as outfile:
        count = 0
        for line in infile:
            count += 1
            img = list(json.loads(line).keys())[0].split(".")[0]
            fea = list(json.loads(line).values())[0]
            fea = np.array(fea)
            cid = nna(fea)
            nnm(rep)
            outfile.write(img + ':' + str(cid) + '\n')
