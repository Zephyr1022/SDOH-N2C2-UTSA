#!/usr/bin/env python3

#!/usr/bin/env python3

# unit of analysis: file -> single dataset 

import glob
import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

input_text = sys.argv[1] #search_data1.yaml
argument_subtype = sys.argv[2]
test_dir = sys.argv[3]
para_data = read_data.ReadData(input_text).loading_data()

def attr_span(type_list,argument_list,dataset):
    attr_dict = {}
    for type_,status in zip(type_list,argument_list): # status: T1 
        attribute  = dataset[status].strip().split('\t')[1].split(' ')[0]
        start = int(dataset[status].strip().split('\t')[1].split(' ')[1])
        end = int(dataset[status].strip().split('\t')[1].split(' ')[-1])
        attr_dict[type_] = [status,attribute,start,end]
        #print("Argument:", status,attribute,start,end)
    
    return attr_dict

def argument_extract(input_file,output_file): 
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    dataset = {}
    sent_list = [] # ALL A1T1E1
    all_data = data.splitlines()
    
    for line in all_data:
        data = line.strip().split()  
        sent_id = data[0] # Entity ID e.g., T1,E1,A1
        sent_list.append(sent_id)
        dataset[sent_id] = line.strip() 
        
    #argument_subtype = 'Type'
        
    all_list_arg = []
    for id_event in sent_list:
        if "E" in id_event:
            argument_list = []
            type_list = []
            
            # trigger_type = Drug; argument_sub = "Type"
            if dataset[id_event].split()[1].split(":")[0] == para_data["event_type"]:  #"Drug": 
                for i in dataset[id_event].split()[2:]:
                    if argument_subtype in i.split(":")[0]:  # argument_sub: # 'Type 'Type2 
                        argument_list.append(i.split(":")[1]) # 'Type:T3', 'Type2:T3' based on E1
                        type_list.append(i.split(":")[0])
                    
                argument = attr_span(type_list,argument_list,dataset) # Event E1 Drug's arguments
                all_list_arg.extend(argument_list) # all the argument entities under "Drug" 
            
    sub_argument = list(set(all_list_arg))
    
    # save ann files
    with open(output_file, "w") as train_trigger:
        for id in sub_argument:
            train_trigger.write(dataset[id]+'\n') 
            

def main():
    
    for filename in glob.glob('./Annotations/'+ test_dir +'/*.ann'): 
        #./Annotations/train/mimic/*.ann' '~/sdoh/Annotations/test/test1' -> ./Annotations/argu_drug
        trigger_filename = filename.replace(test_dir, para_data["argument_file"] +'/test') 
        #print(filename,trigger_filename)
        argument_extract(filename,trigger_filename)
    
main()
