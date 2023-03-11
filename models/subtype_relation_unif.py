#!/usr/bin/env python3

import glob
import pandas as pd

import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

gdtruth_txt = sys.argv[1]
trigger_test = sys.argv[2]
argument_test = sys.argv[3]
subtype_test = sys.argv[4]

def add_tokens(example, triggers, arguments):
    examples = []
    for t in triggers:
        for a in arguments:
            new_string = ''
            check = None
            for i,c in enumerate(example):
                if i != t[1][0] and i != t[1][1] and i != a[1][0] and i != a[1][1]:
                    new_string += c
                else:
                    if i == t[1][0]:
                        if check is None:
                            check = 'trigger_first'
                        new_string += '  <'+t[0]+'>  ' # event1
                    if i == a[1][0]:
                        if check is None:
                            check = 'arg_first'
                        new_string += '  <'+a[0]+'>  ' # argument1
                    if i == a[1][1] and i == t[1][1]:
                        if check == 'trigger_first':
                            if i == a[1][1]:
                                new_string += '  </'+a[0]+'>  ' # argument2
                            if i == t[1][1]:
                                new_string += '  </'+t[0]+'>  ' # event2
                        else:
                            if i == t[1][1]:
                                new_string += '  </'+t[0]+'>  ' # argument2
                            if i == a[1][1]:
                                new_string += '  </'+a[0]+'>  ' # event2
                    else:
                        if i == a[1][1]:
                            new_string += '  </'+a[0]+'>  ' # argument2
                        if i == t[1][1]:
                            new_string += '  </'+t[0]+'>  ' # event2
                    new_string += c
            #print(new_string) # print whole file
            examples.append(new_string.replace("\n", " ")) # new_string.rstrip("\n")
            
    return examples # return a list


def main():
    
    regex = re.compile(r'\d+')
    
    examples = [] # save all prediction sentences
    file_records = []
    all_relations = []
    triggers_exs = []
    args_exs = []
    
    input_file0 = './experiments/system5/piece_relation/'+trigger_test+'_triggers_relation.txt' #triggers
    input_file1 = './experiments/system5/piece_relation/'+argument_test+'_Drug_relation.txt' #Drug 
    input_file2 = './experiments/system5/piece_relation/'+argument_test+'_Alcohol_relation.txt' #Alcohol
    input_file3 = './experiments/system5/piece_relation/'+argument_test+'_Tobacco_relation.txt' #Tobacco
    input_file4 = './experiments/system5/piece_relation/'+argument_test+'_LivingStatus_relation.txt' #LivingStatus
    input_file5 = './experiments/system5/piece_relation/'+argument_test+'_Employment_relation.txt' #Employment
    
    #if there is no event but exist argument? 
    #if there is existing event but no argument?
    
    with open(input_file0, 'r') as iFile_set:
        data_event = iFile_set.read()
        
    with open(input_file1, 'r') as iFile:
        data_drug = iFile.read()
        
    with open(input_file2, 'r') as iFile:
        data_alcohol = iFile.read()
        
    with open(input_file3, 'r') as iFile:
        data_tobacco = iFile.read()
        
    with open(input_file4, 'r') as iFile:
        data_livingStatus = iFile.read()
        
    with open(input_file5, 'r') as iFile:
        data_employment = iFile.read()
        
    #./Annotations/train/mimic/*.ann' 
    for filename in glob.glob(gdtruth_txt): 
        #idx = sorted(list(idx)) # Sort ids 
        idx = regex.findall(filename)
        
        with open(filename, 'r') as iFile:  # txt data
            data = iFile.read()
        
        all_events = data_event.splitlines()
        all_drug_arguments = data_drug.splitlines()
        all_alcohol_arguments = data_alcohol.splitlines()
        all_tobacco_arguments = data_tobacco.splitlines()
        all_livingStatus_arguments = data_livingStatus.splitlines()
        all_employment_arguments = data_employment.splitlines()
        
        # 同一文档，event and argument 存在且相等时，写入
        # 一个 event 5 个 argument 
        for evt_idx in all_events:
#           print(evt_idx.strip().split()[1])
            if evt_idx.strip().split()[0] == idx[0]: # 此文档中events E1 Drug E2 Alcoho
                if evt_idx.strip().split()[1] == 'Drug':
                    for drug_idx in all_drug_arguments:
                        if evt_idx.strip().split()[0] == drug_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(drug_idx.strip().split()[1],(int(drug_idx.strip().split()[2]),int(drug_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                        
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            triggers_exs.append(et[0][0])
                            args_exs.append(arg[0][0])
                        
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(drug_idx.strip().split()[1:4]) + '\t' + ' '.join(drug_idx.strip().split()[4:]))
                        
                elif evt_idx.strip().split()[1] == 'Alcohol':
                    for alcohol_idx in all_alcohol_arguments:
                        if evt_idx.strip().split()[0] == alcohol_idx.strip().split()[0]: #para_data['event_type']: 
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(alcohol_idx.strip().split()[1],(int(alcohol_idx.strip().split()[2]),int(alcohol_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                        
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            triggers_exs.append(et[0][0])
                            args_exs.append(arg[0][0])
                        
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(alcohol_idx.strip().split()[1:4]) + '\t' + ' '.join(alcohol_idx.strip().split()[4:]))
                        
                        
                elif evt_idx.strip().split()[1] == 'Tobacco':
                    for tobacco_idx in all_tobacco_arguments:
                        if evt_idx.strip().split()[0] == tobacco_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(tobacco_idx.strip().split()[1],(int(tobacco_idx.strip().split()[2]),int(tobacco_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                        
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            triggers_exs.append(et[0][0])
                            args_exs.append(arg[0][0])
                        
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(tobacco_idx.strip().split()[1:4]) + '\t' + ' '.join(tobacco_idx.strip().split()[4:]))
                        
                elif evt_idx.strip().split()[1] == 'LivingStatus':
                    for livingStatus_idx in all_livingStatus_arguments:
                        if evt_idx.strip().split()[0] == livingStatus_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(livingStatus_idx.strip().split()[1],(int(livingStatus_idx.strip().split()[2]),int(livingStatus_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                        
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            triggers_exs.append(et[0][0])
                            args_exs.append(arg[0][0])
                        
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(livingStatus_idx.strip().split()[1:4]) + '\t' + ' '.join(livingStatus_idx.strip().split()[4:]))
                        
                else:
                    for employment_idx in all_employment_arguments:
                        if evt_idx.strip().split()[0] == employment_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(employment_idx.strip().split()[1],(int(employment_idx.strip().split()[2]),int(employment_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                        
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            triggers_exs.append(et[0][0])
                            args_exs.append(arg[0][0])
                        
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(employment_idx.strip().split()[1:4]) + '\t' + ' '.join(employment_idx.strip().split()[4:]))
                        
                        
    list_match = ['N/A' for i in range(len(examples))] # unknown for test
    
    c = {"sentence": examples,
        "matchorNot": list_match,
        "ann": file_records,
        "relation": all_relations,
        "event": triggers_exs,
        "argument": args_exs
    }
    
    df = pd.DataFrame(c)

    options_med = ['Drug', 'Alcohol','Tobacco']
    df_med = df[df["event"].isin(options_med) & (df["argument"] == "StatusTime")]
    
    options_emp = ['Employment']
    df_emp = df[df["event"].isin(options_emp) & (df["argument"] == "StatusEmploy")]
    
    options_liv = ['LivingStatus']
    df_liv_Status = df[df["event"].isin(options_liv) & (df["argument"] == "StatusTime")] 
    df_liv_Type = df[df["event"].isin(options_liv) & (df["argument"] == "TypeLiving") ]
    
    
    # Exporting the DataFrame into a CSV file
    #df.to_csv('relation_test.csv', header = False) # relative position relation_dev.csv
    df_med.to_csv('./experiments/system5/piece_subtype/subtype_'+subtype_test+'_med.csv', header = False) # relative position
    df_emp.to_csv('./experiments/system5/piece_subtype/subtype_'+subtype_test+'_emp.csv', header = False) # relative position
    df_liv_Status.to_csv('./experiments/system5/piece_subtype/subtype_'+subtype_test+'_liv_status.csv', header = False) # relative position
    df_liv_Type.to_csv('./experiments/system5/piece_subtype/subtype_'+subtype_test+'_liv_type.csv', header = False) # relative position
    
    print(len(examples))
    print(len(df_med),len(df_emp),len(df_liv_Status),len(df_liv_Type))
    
main()
