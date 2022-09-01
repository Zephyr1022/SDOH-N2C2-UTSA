#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, sys, re
import funztools

all_poss_data = sys.argv[1]
match_prob = sys.argv[2]
output_match = sys.argv[3] # output 命名
threshold = sys.argv[4]

regex = re.compile(r'\d+') #probs = regex.findall(row[1])

with open(all_poss_data, 'r') as iFile_relation: #"event-argument-all-poss-relation.txt"
    relation = iFile_relation.read()
relation_ = relation.splitlines()
        
with open(match_prob) as iFile: #'./relation_pred/test-base-match-prob-V2.csv'
    data_prob = iFile.read()
all_data_prob = data_prob.splitlines() # check - handle data
probs = []
for i,row in enumerate(all_data_prob):
    probs.append(row.strip().replace('tensor([','').replace('])',''))
    
with open(output_match, "w") as match_trigger: # "event-argument-match-relation.txt"
    for i, (line, rel) in enumerate(zip(probs,relation_)):
        p = line.strip().split(',')
        r = rel.strip().split(' ')
        match_prob = float(p[0]) #notmat_p = float(a[1])
        key = ''.join(r[:4])
        
        if match_prob > float(threshold):
            #print(rel,match_p)
            match_trigger.write(rel)
            match_trigger.write('\n')     
            
            
'''
from collections import defaultdict

data = defaultdict(list)

data[(1109, 'LivingStatus', 41, 46)].append(('StatusTime', 41, 46, .992))
data[(1109, 'LivingStatus', 41, 46)].append(('StatusTime', 50, 80, .733))
data[(1, 'LivingStatus', 41, 46)].append(('StatusTime', 41, 46, .11))
data[(1, 'LivingStatus', 41, 46)].append(('StatusTime', 50, 80, .3333))
data[(1, 'LivingStatus', 41, 46)].append(('StatusTime', 50, 92, .01))

for key,v in data.items(): # Argmax
    max_tuple = None
    for v2 in v:
        if (max_tuple is None or v2[3] > max_tuple[3]) and v2[3] > 0.1:
            max_tuple = v2
    print(key, max_tuple,max_tuple[3])
    
'''
