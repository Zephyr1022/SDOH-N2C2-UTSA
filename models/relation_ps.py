#!/usr/bin/env python3

# unit of analysis: file -> single dataset 
# python relation_ps.py 

import glob
import pandas as pd

import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

train_dev_test_dir = sys.argv[1]
#input_text = sys.argv[1] #search_data1.yaml
#para_data = read_data.ReadData(input_text).loading_data()

'''
Example: The outdoors <Drug> <type> Retired <\type> assistant to nutrition< <\Drug>.

'''

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
                    
            if t[1][1] == len(example):
                new_string += '  </'+t[0]+'>  ' # event2
                #print("found it")
            if a[1][1] == len(example):
                new_string += '  </'+a[0]+'>  ' # argument2
            #print("found it")
            
            #print(new_string) # print whole file
            examples.append(new_string.replace("\n", " ")) # new_string.rstrip("\n")
            #print(examples)
            
    return examples # return a list


def attr_span(argument_type,argument_list,dataset):
    
    attr_dict = {}
    for type_,status in zip(argument_type,argument_list): # status: T1 
    
        attribute  = dataset[status].strip().split('\t')[1].split(' ')[0]
        start = int(dataset[status].strip().split('\t')[1].split(' ')[1])
        end = int(dataset[status].strip().split('\t')[1].split(' ')[-1])
    
        attr_dict[type_] = [status,attribute,start,end]
        #print("Argument:", status,attribute,start,end)
    
    return attr_dict # return a dict

def event_argument(input_file): 
    
    # build dictionary
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    dataset = {}
    sent_list = [] # all the A1T1E1

    all_data = data.splitlines()
    for line in all_data:
        data = line.strip().split()  
        sent_id = data[0] # Entity ID e.g., T1,E1,A1
        sent_list.append(sent_id)
        dataset[sent_id] = line.strip() 
        
    # Five Type of Events 
    event_types = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
    
    all_list_arg = []
    all_list_evt = []
    
    for evtp in event_types:
        #print(evtp)
        relations = []
        argument_out = []
        event_out  = []
        for id_event in sent_list:
            
            # Loop Events(E1,E2,...): E1 Drug
            if "E" in id_event: 
                event_type = [] # Drug
                event_list = []
                argument_list = []
                argument_type = []
                relation = [] # [[[drug1],[argument]] [drug2,argument]]
                #print(id_event,evtp)
                
                if dataset[id_event].split()[1].split(":")[0] == evtp:
                    #print("test", id_event, dataset[id_event].split()[1].split(":")[0])     # test E3 Drug
                    
                    event_list.append(dataset[id_event].split()[1].split(":")[1])
                    event_type.append(dataset[id_event].split()[1].split(":")[0])
                    #print('test2', event_list,event_type)
                    
                    for i in dataset[id_event].split()[2:]:
                        argument_list.append(i.split(":")[1])# 'Status:T2', 'Type:T3', 'Type2:T3' based on E1
                        argument_type.append(i.split(":")[0]) #argument_type
                        
                    event = attr_span(event_type,event_list,dataset)
                    argument = attr_span(argument_type,argument_list,dataset) # Event E1 Drug's arguments
                    
                    argument_in = []
                    for key in event: # inside loop Ei key is fix
                        et = [[(event[key][1],(event[key][2],event[key][3]))]]
                        #print('test1',et)
                    
                    for key2 in argument:
                        arg = [(argument[key2][1],(argument[key2][2],argument[key2][3]))]
                        argument_in.append(arg)
                        
                    event_out.append(et)
                    argument_out.append(argument_in)
                    
        all_list_evt.extend(event_out)
        all_list_arg.extend(argument_out)
        
    return all_list_evt,all_list_arg # two set event-argument 

    
def main():
    
    examples = [] # save all train sentences
    file_records = []
    regex = re.compile(r'\d+')
    
    # './Annotations/dev/mimic/*.ann'
    #trigger_filename = filename.replace('./Annotations/'+para_data['argument_train_dev']+'/mimic', "./Annotations/dev/test") # no use 
        #input (filename) -> output (trigger_filename)
        #print('\ntrigger_filename:', txt_filename)
    
    #for filename in glob.glob('./Annotations/'+para_data['argument_train_dev']+'/mimic/*.ann'): #./Annotations/train/mimic/*.ann' 
    for filename in glob.glob('./Annotations/'+ train_dev_test_dir +'/temp/*.ann'):
        txt_filename = filename.replace(".ann", ".txt")
        idx = regex.findall(filename)
    
        events,arguments = event_argument(filename)     # input_file,output_file no use 
        # all relations b/t drug and arguments
        #print(events[7],arguments[7])  # extract the corresponding event trigger and arguments
    
        relations = [] 
        for i,c in enumerate(events):
            for j,d in enumerate(arguments):
                if i == j: 
                    for l in arguments[j]:
                        relation = []
                        relation.append(events[i][0])
                        relation.append(l)
                        relations.append(relation)
                        
        #print(relations)
                        
        with open(txt_filename, 'r') as iFile: # open file -> text data
            data = iFile.read()
    
            for rl in relations:
                #print('pair:',rl)
                triggers_ex = rl[0]   #[('Drug', (243,252))]
                args_ex = rl[1]   #[('StatusTime', (293,307))]
                example = add_tokens(data,triggers_ex,args_ex)
                examples.append(example[0])
                #file_records.append(txt_filename.replace('./Annotations/'+para_data['argument_train_dev']+'/mimic/', ""))
                file_records.append(idx[0])
                #print('\n')
                
    list_match = ['match' for i in range(len(examples))]
    
    c = {"sentence":examples,
        "matchorNot": list_match,
        "ann": file_records}
    df = pd.DataFrame(c)
    
    # Exporting the DataFrame into a CSV file
    df.to_csv('relation_'+ train_dev_test_dir +'_match.csv', header = False) # relative position
    print(len(examples))
    
main()
