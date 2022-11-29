import itertools
import pandas as pd
from collections import defaultdict
from itertools import islice, chain
import re
from datetime import datetime
from datetime import timedelta
import networkx as nx
import numpy as np
import os
import dateutil.parser
import scipy.sparse as sp


def tocsv(data,path):
    data_g = pd.DataFrame(data)
    print(data_g)
    data_g.to_csv(path, header=None, index=None, sep='\t', mode='a')


# make negtive and positive lable 
def train_test_preprocess(path):

    data = np.genfromtxt(path, dtype=int, delimiter='\t', encoding=None)

    data_vers = data[:, [1,0]]
    new_data = np.vstack((data,data_vers))  # for undirction kg 
    uni_data = np.unique(new_data,axis=0)   # if not de-deduplication, will has value of 2 or -1 in adj matrix

    l = uni_data[:,0]
    r = uni_data[:,1]
    nodes = np.hstack((l,r))
    node2id = {k: v for v, k in enumerate(list(set(nodes)))}
    id2node =  {k: v for k, v in enumerate(list(set(nodes)))}

    number_of_nodes = len(node2id)
    print('number of all nodes',number_of_nodes)

    data_id = []
    u,v = [],[]
    for i in range(len(uni_data)):
        a = node2id[uni_data[i,0]]
        b = node2id[uni_data[i,1]]
        u.append(a)
        v.append(b)
        data_id.append([a,b])

    return node2id, id2node, u, v, number_of_nodes


def neg_sample(u,v,number_of_nodes):
    '''
    build negtive sample nodes (no dirction)
    '''

    adj = sp.coo_matrix((np.ones(len(u)), (u, v)))
    # print(adj.shape)

    adj_neg = 1 - adj.todense() - np.eye(max(u)+1,max(v)+1)
    neg_u_full, neg_v_full = np.where(adj_neg != 0)

    return neg_u_full, neg_v_full
    # neg = []
    # for i in range(len(neg_u_full)):
    #     neg.append([neg_u_full[i],neg_v_full[i]])
    # neg_data = []

def build_train_test(u,v,neg_u,neg_v,id2node):

    eids = np.arange(len(u)) # 0-len(data)
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.75)

    # positive 
    train_size = len(data_id) - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]] # 253
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]# 85
    # print('pos test',len(test_pos_u))
    # print('pos train',len(train_pos_u))

    # negtive
    neg_eids = np.random.choice(len(neg_u), len(u)) # choice len(u) number of values from 0-len(neg_u) 
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    # print('neg test',len(test_neg_u))
    # print('neg train',len(train_neg_u))

    # store
    # positive edges of val set
    # edge[2] == 1 pos
    # edge[2] == 0 neg
    # edge[3] == 0 val 20%
    # edge[3]  == 1 train 80% * 25%
    # edge[3]  == 2 test 80% * 75% 
    data = []
    for i in range(len(test_pos_u)):
        data.append((id2node[test_pos_u[i]],id2node[test_pos_v[i]],1,2))
        data.append((id2node[test_neg_u[i]],id2node[test_neg_v[i]],0,2))

    for i in range(len(train_pos_u)):
        data.append((id2node[train_pos_u[i]],id2node[train_pos_v[i]],1,1))
        data.append((id2node[train_neg_u[i]],id2node[train_neg_v[i]],0,1))

    return data


node2id, id2node, u, v, number_of_nodes = train_test_preprocess('train_test_id_4w.txt')
u_array = np.array(u)
v_array = np.array(v)
neg_u, neg_v = neg_sample(u_array,v_array,number_of_nodes)
train_test_val = build_train_test(u_array,v_array,neg_u, neg_v,id2node)

# print(train_test_val)

# tocsv(train_test_val,'load_tran_test_4w.txt')


