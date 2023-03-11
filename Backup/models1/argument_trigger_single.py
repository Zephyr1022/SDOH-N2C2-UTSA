#!/usr/bin/env python3

#!/usr/bin/env python3

# unit of analysis: file -> single dataset 

# python argument_trigger.py sdoh-26-drug.yaml > result.out 2>&1 &

import glob
import os, sys
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

input_text = sys.argv[1] #search_data1.yaml
argument_subtype = sys.argv[2]
para_data = read_data.ReadData(input_text).loading_data()

# find overlap 2x2
def overlap(start1, end1, start2, end2):
    """Does the range (start1, end1) overlap with (start2, end2)?"""
    return (
        start1 <= start2 <= end1 or
        start1 <= end2 <= end1 or
        start2 <= start1 <= end2 or
        start2 <= end1 <= end2
    )
    
    
def attr_span(type_list,argument_list,dataset):
    
    attr_dict = {}
    for type_,status in zip(type_list,argument_list): # status: T1 
    
        attribute  = dataset[status].strip().split('\t')[1].split(' ')[0]
        start = int(dataset[status].strip().split('\t')[1].split(' ')[1])
        end = int(dataset[status].strip().split('\t')[1].split(' ')[-1])
    
        attr_dict[type_] = [status,attribute,start,end]
        #print("Argument:", status,attribute,start,end)
    
    return attr_dict

def count_freq(diff_list, del_overlap_list,dataset):
    for dl in  diff_list:
        for ol in del_overlap_list:
            #print(dl,ol) T21    History 375 401    until recent detox program
            dl_type = dataset[dl].split()[1].split(":")[0]
            dl_stat = dataset[dl].split()[2].split(":")[0]
            dl_end = dataset[dl].split()[3].split(":")[0]
            
            ol_type = dataset[ol].split()[1].split(":")[0]
            ol_stat = dataset[ol].split()[2].split(":")[0]
            ol_end = dataset[ol].split()[3].split(":")[0]
            
            if overlap(dl_stat,dl_end,ol_stat,ol_end):
                print(ol_type,dl_type)
                
def single_argument(input_file,output_file): 
    
    # build dictionary
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    dataset = {}
    sent_list = [] # all the A1T1E1
    for line in data.splitlines():
        data = line.strip().split()  
        sent_id = data[0] # Entity ID e.g., T1,E1,A1
        sent_list.append(sent_id)
        dataset[sent_id] = line.strip() 
        
        
    # delect overlap
    del_overlap_list = []
    # count overlap freq
    all_list_arg = []
    
    for id_event in sent_list:
        if "E" in id_event: # Loop Events(E1,E2,...): E1 Drug
            #event_type.append(data[1].split(":")[0])
            #event_list.append(data[1].split(":")[1])
            argument_list = []
            type_list = []
        
            if dataset[id_event].split()[1].split(":")[0] == para_data["event_type"]: #"Drug": # event type; argument_sub = "Status"
                #print(dataset[id_event].split()[2:])
                for i in dataset[id_event].split()[2:]:
                    
                    if argument_subtype in i.split(":")[0]: # argument_sub: #是否是 Status 'Type 'Type2 
                        argument_list.append(i.split(":")[1])# 'Status:T2', 'Type:T3', 'Type2:T3' based on E1
                        type_list.append(i.split(":")[0])
                    
                argument = attr_span(type_list,argument_list,dataset) # Event E1 Drug's arguments
                #print(argument)
                all_list_arg.extend(argument_list) # all the argument entities under "Drug" 
        
        
                key_list = [] # detect overlap
                key_list_sp = []
        
                if "Status" in argument:
                    del_overlap_list.append(argument['Status'][0])
                    for key in argument: # key should be unique
                        #print(key)
                    
                        if overlap(argument['Status'][2], argument['Status'][3], argument[key][2], argument[key][3]):
                            continue
                        else:
                            key_list.append(key) # list exclude 'Status' within E1, itself add
                            #print('key_list:',key_list)
                            
                            overlap_ = 0
                            for k in key_list:
                                if k != key:
                                    overlap_ += overlap(argument[k][2], argument[k][3], argument[key][2], argument[key][3])+0
                                    #print('compare:','key:',key,k,overlap_)
                                    
                            if overlap_== 0:
                                del_overlap_list.append(argument[key][0])
                                #print("test",argument[key][0])
                                
                else: # missing "Status"
                    for key in argument: # key should be unique
                        #print('missing key:',key)
                        key_list_sp.append(key)
        
                        overlap_ = 0
                        for k in key_list_sp:
                            if k != key:
                                overlap_ += overlap(argument[k][2], argument[k][3], argument[key][2], argument[key][3])+0
                                    #print('compare:','key:',key,k,overlap_)
                                
                        if overlap_== 0:
                                del_overlap_list.append(argument[key][0])
                            
                            
                #print("test overlap:", key_list_sp)
    del_overlap_list = set(del_overlap_list)
    del_overlap_list = list(del_overlap_list)
    #print(del_overlap_list)
    diff_list = list(set(all_list_arg)-set(del_overlap_list))
    count_freq(diff_list,del_overlap_list,dataset) # dataset entities dict
    
    sub_argument = list(set(all_list_arg))
    #print(sub_argument)
        
    # save ann files
    with open(output_file, "w") as train_trigger:
        for id in del_overlap_list:
            '''
            train_trigger.write(str(id)+"\t")
            train_trigger.write(dataset[id][0]+" "+ dataset[id][1]+ " "+ dataset[id][2]+'\t')
            train_trigger.write(' '.join(dataset[id][3:]))
            '''
            #print(dataset[id])
            
            train_trigger.write(dataset[id]+'\n') 
            #train_trigger.write(dataset[id].strip().split()[0]+"\t"+para_data["event_type"]+"-Related"+" "
            #            +dataset[id].strip().split()[2]+" "+dataset[id].strip().split()[3]
            #            +"\t"+' '.join(dataset[id].strip().split()[4:])+'\n')
            
            
            
def main():
    
    #print(para_data["argument_train_dev"])
    print(para_data["argument_file"],para_data["argument_train_dev"])
    
    for filename in glob.glob('./Annotations/'+ para_data["argument_train_dev"] +'/mimic/*.ann'): #./Annotations/train/mimic/*.ann'
    
        trigger_filename = filename.replace('/Annotations/'+ para_data["argument_train_dev"] +'/mimic', '/Annotations/'+ para_data["argument_file"] +'/'+ para_data["argument_train_dev"]) 
    
        #print(filename,trigger_filename)
        single_argument(filename,trigger_filename)
    
main()