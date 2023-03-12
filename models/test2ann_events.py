#!/usr/bin/env python3

import os, sys
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

#input_text = sys.argv[1] #search_data1.yaml
#para_data = read_data.ReadData(input_text).loading_data()

ann_dir = sys.argv[1] # filename = ann_dir + f +'.ann' # './Annotations/test/test1_table/' event table 坯子
conll_order = sys.argv[2] # 对应的 'test_events_num.conll'
unorder_pred = sys.argv[3] # 之前一步 predict 'test_events_pred1.txt' 
output_pred = sys.argv[4] # 'events_relation_pred1.txt' 命名

def conll_predict(input_file, input_file2, output_file):

    with open(input_file, 'r') as iFile:
        data = iFile.read()
    with open(input_file2, 'r') as iFile2:
        data2 = iFile2.read()
        
    sent = []
    word = []
    dataset = {}
    sent_id = 0
    idx = 0
    
    data_all = data.splitlines()
    
    for line in data_all: # Get all IDs
        idx+=1
        if len(line.strip().split()) == 0:
            # print(sent_id)
            dataset[sent_id] = word
            idx = 0
            sent_id += 1 
            word = []
            continue
   
        newList = line.strip().split()[0:2]+line.strip().split()[3:6]
        #print(newList)
        word.append(newList)
    ### THIS IS A TEST
    # print(sent_id)
    dataset[sent_id] = word
    idx = 0
    sent_id += 1 
    word = []
   
    newList = line.strip().split()[0:2]+line.strip().split()[3:6]
    #print(newList)
    word.append(newList)
    #####################
  
    
    data2_all = data2.splitlines()
    #print(data2_all)
    with open(output_file, "w") as pred_trigger:
        for line in data2_all:
            data = line.strip().split()

            snt = int(data[0])
            chr_str = int(data[2].split(",")[0])
            chr_end = int(data[2].split(",")[-1])
            
            print(data)
            
            if chr_str != chr_end:
                pred_trigger.write(dataset[snt][chr_str-1][-1]+' '+ data[1] +' '+ dataset[snt][chr_str-1][2] +' '+ dataset[snt][chr_end-1][3] +' '+ ' '.join(data[3:]) +' '+'\n')
                
            else:
                pred_trigger.write(dataset[snt][chr_str-1][-1]+' '+ data[1] + ' '+ ' '.join(dataset[snt][chr_str-1][2:4]) +' '+ ' '.join(data[3:])+' '+'\n')
    
                
def add_event_trigger(input_file): # deal with single file
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    with open(input_file, "a") as add_trigger: # add more detial not rewrite
        data3_all = data.splitlines()
        for line in data3_all: 
            if len(line.strip().split()) == 0:
                continue
            add_trigger.write("E"+line.strip().split()[0][1]+"\t")
            add_trigger.write(line.strip().split()[1]+":")
            add_trigger.write(line.strip().split()[0])
            add_trigger.write('\n')
            #print("E" + line.strip().split()[0][1] + '\t'+line.strip().split()[1]+':'+line.strip().split()[0])


def ann_split(input_file1,input_file2):
    
    with open(input_file1, 'r') as iFile_set:
        data_idx = iFile_set.read()
        
    with open(input_file2, 'r') as iFile:
        data = iFile.read()
        
    files = set()
    data_idx_all = data_idx.splitlines()
    
    for line in data_idx_all: # Get all IDs
        if len(line.strip().split()) == 0:
            continue
        file_id = line.strip().split()[-1]
        files.add(file_id) 
    
    for f in files:
        filename = ann_dir + f +'.ann' # './Annotations/events/test/f.ann'
        print(filename)
        with open(filename, "w") as split_trigger:
            i = 0
            data4_all = data.splitlines()
            for line in data4_all: # Get all IDs
                if len(line.strip().split()) == 0:
                    continue
                if line.strip().split()[0] == f:
                    i += 1 
                    split_trigger.write("T"+str(i)+"\t")
                    split_trigger.write(' '.join(line.strip().split()[1:4])+"\t")
                    split_trigger.write(' '.join(line.strip().split()[4:]))
                    split_trigger.write('\n')
        
def main():
    
    conll_predict(conll_order,unorder_pred,output_pred)
    ann_split(conll_order,output_pred)  
    
main()
