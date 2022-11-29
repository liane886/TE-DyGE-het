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
import json


def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    date = datetime.strftime(d,'%Y%m%d')
    date_M = d.strptime(date,'%Y%m%d')
    return date_M

def build_links(kg_df):    
    links = []
    ts = []
    for i in range(len(kg_df)):
        timesting = kg_df['time'][i]
        timestamp = getDateTimeFromISO8601String(timesting)
        ts.append(timestamp)
        links.append((kg_df['head_code'][i],kg_df['relation'][i],kg_df['tail_code'][i],timestamp))
    return links,ts

def toscv(data,path):
    data_g = pd.DataFrame(data)
    print(data_g)
    data_g.to_csv(path, header=None, index=None, sep='\t', mode='a')

def save_id(file,path):
    json_str = json.dumps(file, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)

def pre_process(path):
    data = pd.read_csv(path)
    kg = data.drop(['Unnamed: 0'],axis =1)
    kg = kg.drop(kg[kg['relation']=='unknown'].index)
    kg= kg.sort_values(['time'])
    kg = kg.reset_index()
    return kg



def build_relation2id(kg):
    relation_ls = kg['relation'].unique()
    relation2id = {}
    for i in range(len(relation_ls)):
        relation2id[relation_ls[i]] = i
    return relation2id


def build_val(relation2id,links,ts, SLICE_WEEKS):
    '''
        Gtp1: G(t+1) subgraph nodes
        Gt: G0 - Gt subgraph nodes
    '''
    #### build val dataset G(t+1)
    Gtp1 = []
    Gt = []
    company2id = {}
    START_DATE = min(ts)
    MAX_DATE = max(ts)
    print ("Start date", START_DATE)
    print('End daye',MAX_DATE)
    slice_id = 0
    count = 0
    for (a, b, c, time) in links:
        if a not in company2id:
            company2id[a] = count
            count += 1
        if c not in company2id:
            company2id[c] = count 
            count += 1
        strtime = datetime.strftime(time,'%Y%m%d')

        # find the nodes in G(t+1) subgraph
        if (MAX_DATE - time).days <= 7*SLICE_WEEKS:
            Gtp1.append((company2id[a],company2id[c],relation2id[b],strtime))
           
        else:
            Gt.append((company2id[a],company2id[c],relation2id[b],strtime))
    return company2id, Gt, Gtp1


def process_Gtp1(Gtp1,val_p):
    '''
        deal with the G(t+1) subgraph
    '''
    edges_val = []
    for (a,c,b,time) in Gtp1:
        if (a,c) not in edges_val:
            edges_val.append((a,c))
    val_size = int(len(edges_val) * val_p)
    # train_size = int(len(edges_val) * val_p * train_p)
    val = edges_val[:val_size]
    val_els = edges_val[val_size:]
    print('# val nodes',len(edges_val))
    print('# train and test in validation',len(val_els))

    return val, val_els


kg = pre_process('../../FinKG/data/company_relation.csv')
links,ts = build_links(kg)
links_dedue  = list(set(links))
relation2id = build_relation2id(kg)
SLICE_WEEKS = 4
company2id, Gt, Gtp1 = build_val(relation2id,links_dedue,ts,SLICE_WEEKS)
val_p = 0.2
val,tran_test_validation = process_Gtp1(Gtp1,val_p)

toscv(links_dedue,'../../FinKG/alldata_processed.txt')

toscv(tran_test_validation,"../../FinKG/train_test_id_4w.txt")
toscv(val,"../../FinKG/val_id_4w.txt")

save_id(company2id,'../../FinKG/company2id.txt')
save_id(relation2id,'../../FinKG/relation2id.txt')




# company str2id 
# modify the data_helper2, read_dynamic_graph,  load validation edges
# make a link from company name to id to id_map(from data helper)

# relation_ls = kg['relation'].unique()
# relation2id = {}
# for i in range(len(relation_ls)):
#     relation2id[relation_ls[i]] = i

# company2id = {}
# count = 0
# graph = []
# for (a,b,c,time) in links:
#     if a not in company2id:
#         company2id[a] = count
#         count += 1
#     if c not in company2id:
#         company2id[c] = count 
#         count += 1

    # trip = dict()
    # trip['Node_one'] = company2id[a]
    # trip['Node_two'] = company2id[c]
    # trip['Edge_type'] = relation2id[b]
    # trip['Timestamp'] = time

    # if trip not in graph:
    #     graph.append(trip)


# graph_d = pd.DataFrame(graph)

# graph_d.to_csv('KQ_train2.txt', header=None, index=None, sep='\t', mode='a')

# company_id = json.dumps(company2id,indent=2)
# with open('xompany2id.json','w') as json_file:
#     json_file.write(company_id)
