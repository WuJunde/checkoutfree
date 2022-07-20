from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.pygcn.utils import load_data, accuracy
from models.pygcn.models import GCN
import cfg
import torch
from torch import nn

args = cfg.parse_args()
def pick_pairs(labels):
    pairs = []
    mode = 'tri'
    for ind, lab in enumerate(labels):
        if lab == -1:
            continue
        else:
            posind = (labels == lab).nonzero(as_tuple=False)
            negind = ((labels != lab) & (labels != -1)).nonzero(as_tuple=True)
            # negind = t[t!=(labels == -1)]
            for i in range(min(len(posind), len(negind))):
                pairs.append((ind, posind[i], negind[i]))
    if len(pairs) == 0:
        vid_labs_ind = (labels != -1).nonzero(as_tuple=False)
        if len(vid_labs_ind) < 2:
            return pairs, 'None'
        for i, ind in enumerate(vid_labs_ind):
            for nind in vid_labs_ind[i+1:]:
                pairs.append((ind, nind))
        if labels[vid_labs_ind[0]] == labels[vid_labs_ind[1]]:
            mode = 'pos'
        else:
            mode = 'neg'

    return pairs, mode

def train(epoch, model, optimizer, features, adj, labels):
    t = time.time()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    cos_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)
    loss_train = 0
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    tpairs,mode = pick_pairs(labels)
    if len(tpairs) == 0:
        print('all None graph, skip')
        return
    if mode == 'tri':
        for (aind, pind, nind) in tpairs:
            loss_train += triplet_loss(output[aind], output[pind], output[nind])
    elif mode == 'pos':
        for (ind1, ind2) in tpairs:
            loss_train -= cos_loss(output[ind1], output[ind2])
    elif mode == 'neg':
        for (ind1, ind2) in tpairs:
            loss_train += cos_loss(output[ind1], output[ind2])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    return output

def test(model, optimizer, features, adj, labels):
    model.eval()
    loss_test = 0
    output = model(features, adj)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    cos_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)
    tpairs,mode = pick_pairs(labels)
    if len(tpairs) == 0:
        print('all None graph, skip')
        return
    if mode == 'tri':
        for (aind, pind, nind) in tpairs:
            loss_test += triplet_loss(output[aind], output[pind], output[nind])
    elif mode == 'pos':
        for (ind1, ind2) in tpairs:
            loss_test -= cos_loss(output[ind1], output[ind2])
    elif mode == 'neg':
        for (ind1, ind2) in tpairs:
            loss_test += cos_loss(output[ind1], output[ind2])
    ave_loss = loss_test / len(tpairs)
    print("loss= {:.4f}".format(ave_loss.item()))
    return output

# Train model
def run_gcn(args,csg_per, link_map):
    # Training settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(csg_per, link_map)

    # Model and optimizer
    model = GCN(nfeat=256,
                nhid=args.hidden,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    if args.weights:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        state_dict = checkpoint['state_dict']
        if args.distributed:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k[7:] # remove `module.`
                name = 'module.' + k
                new_state_dict[name] = v
            # load params
        else:
            new_state_dict = state_dict

        model.load_state_dict(new_state_dict)

    if args.mode == 'train':
        t_total = time.time()
        for epoch in range(args.epochs):
            output = train(epoch,model, optimizer, features, adj, labels)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    elif args.mode == 'test':
        # Testing
        output = test(model, optimizer, features, adj, labels)
    else:
        assert('mode cannot recognized')
    return output
