#!/usr/bin/env python3
import pandas as pd
import numpy as np
import funztools
import os, sys, re

all_poss_data = sys.argv[1]
match_prob = sys.argv[2]
output_match = sys.argv[3] # output 命名
threshold = sys.argv[4]

regex = re.compile(r'\d+') #probs = regex.findall(row[1])

with open(all_poss_data, 'r') as iFile_relation:
    relation = iFile_relation.read()
relation_ = relation.splitlines()
        
with open(match_prob) as iFile:
    data_prob = iFile.read()
all_data_prob = data_prob.splitlines() # check - handle data
probs = []
for i,row in enumerate(all_data_prob):
    probs.append(row.strip().replace('tensor([','').replace('])',''))
    

from collections import defaultdict

# key: 2724 LivingStatus 16 21 TypeLiving

data = defaultdict(list)
for i, (line, rel) in enumerate(zip(probs,relation_)):
    p = line.strip().split(',')
    r = rel.strip().split(' ')
    match_prob = float(p[0])
    if match_prob > float(threshold):
        key = (r[0],r[1],r[2],r[3],r[4])
        value = (r[5],r[6],match_prob,rel)
        data[key].append(value)

    
with open(output_match, "w") as match_trigger:
    for key,v in data.items(): # Argmax
        max_tuple = None
        for v2 in v:
            #print(v2[-1].strip().split()[4])
            if 'Status' in v2[-1].strip().split()[4]:
                if (max_tuple is None or v2[2] > max_tuple[2]): #and v2[2] > float(threshold)
                    max_tuple = v2
            else:
                max_tuple = v2

        #print(max_tuple[-1])    
        match_trigger.write(max_tuple[-1])
        match_trigger.write('\n')
        
        #print(key, max_tuple[-1],max_tuple[2])
