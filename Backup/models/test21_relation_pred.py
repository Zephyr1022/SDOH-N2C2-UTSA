import glob
import pandas as pd

import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

#input_text = sys.argv[1] #search_data1.yaml
groundtruth_txt = sys.argv[1]
output_csv = sys.argv[2]
all_poss_rel_dir = sys.argv[3]
#para_data = read_data.ReadData(input_text).loading_data()


def write_relation(input_list): 
    all_relation = input_list 
    with open(all_poss_rel_dir, "w") as event_table: #"event-tag-argument-all-poss-relation.txt"
        for l in all_relation:
            #print(l)
            event_table.write(l)
            event_table.write('\n')
            

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
    
    input_file0 = './test_relation/test2_triggers_relation.txt' #Trigger
    input_file1 = './test_relation/test1_Drug_relation.txt' #Drug 
    input_file2 = './test_relation/test1_Alcohol_relation.txt' #Alcohol
    input_file3 = './test_relation/test1_Tobacco_relation.txt' #Tobacco
    input_file4 = './test_relation/test1_LivingStatus_relation.txt' #LivingStatus
    input_file5 = './test_relation/test1_Employment_relation.txt' #Employment
    
    
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
    for filename in glob.glob(groundtruth_txt): 
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
            if evt_idx.strip().split()[0] == idx[0]: # 此文档中events E1 Drug E2 Alcoho
                if evt_idx.strip().split()[1] == 'Drug':
                    for drug_idx in all_drug_arguments:
                        if evt_idx.strip().split()[0] == drug_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(drug_idx.strip().split()[1],(int(drug_idx.strip().split()[2]),int(drug_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                            
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(drug_idx.strip().split()[1:4]) + '\t' + ' '.join(drug_idx.strip().split()[4:]))
                            
                elif evt_idx.strip().split()[1] == 'Alcohol':
                    for alcohol_idx in all_alcohol_arguments:
                        if evt_idx.strip().split()[0] == alcohol_idx.strip().split()[0]: #para_data['event_type']: 
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(alcohol_idx.strip().split()[1],(int(alcohol_idx.strip().split()[2]),int(alcohol_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                            
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(alcohol_idx.strip().split()[1:4]) + '\t' + ' '.join(alcohol_idx.strip().split()[4:]))
                            
                            
                elif evt_idx.strip().split()[1] == 'Tobacco':
                    for tobacco_idx in all_tobacco_arguments:
                        if evt_idx.strip().split()[0] == tobacco_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(tobacco_idx.strip().split()[1],(int(tobacco_idx.strip().split()[2]),int(tobacco_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                            
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(tobacco_idx.strip().split()[1:4]) + '\t' + ' '.join(tobacco_idx.strip().split()[4:]))
                            
                elif evt_idx.strip().split()[1] == 'LivingStatus':
                    for livingStatus_idx in all_livingStatus_arguments:
                        if evt_idx.strip().split()[0] == livingStatus_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(livingStatus_idx.strip().split()[1],(int(livingStatus_idx.strip().split()[2]),int(livingStatus_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                            
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(livingStatus_idx.strip().split()[1:4]) + '\t' + ' '.join(livingStatus_idx.strip().split()[4:]))
                            
                else:
                    for employment_idx in all_employment_arguments:
                        if evt_idx.strip().split()[0] == employment_idx.strip().split()[0]: #para_data['event_type']: 'Drug'
                            et = [(evt_idx.strip().split()[1], (int(evt_idx.strip().split()[2]),int(evt_idx.strip().split()[3])))]
                            arg = [(employment_idx.strip().split()[1],(int(employment_idx.strip().split()[2]),int(employment_idx.strip().split()[3])))]   
                            example = add_tokens(data,et,arg) # txt_file, event, argument: [('Drug', (243, 252))] [('Type', (243, 252))]
                            
                            examples.append(example[0]) #str
                            file_records.append(idx[0])
                            
                            all_relations.append(" ".join(evt_idx.strip().split()[:4]) +' '+' '.join(employment_idx.strip().split()[1:4]) + '\t' + ' '.join(employment_idx.strip().split()[4:]))

                            
    list_match = ['match' for i in range(len(examples))] # unknown for test
    
    c = {"sentence": examples,
        "matchorNot": list_match,
        "ann": file_records,
        "relation": all_relations}
    
    df = pd.DataFrame(c)
    write_relation(all_relations)
    print(len(examples))
    
    # Exporting the DataFrame into a CSV file
    df.to_csv(output_csv, header = False) # relative position relation_dev.csv
    
main()   
