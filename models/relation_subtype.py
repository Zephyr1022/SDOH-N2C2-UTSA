# unit of analysis: file -> single dataset 

import glob
import pandas as pd

import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

input_text = sys.argv[1] #search_data1.yaml
para_data = read_data.ReadData(input_text).loading_data()

def add_tokens(example, triggers, arguments):
    examples = []
    for t in triggers:
        #print(t[1][0],t[1][1],len(example))
        for a in arguments:
            new_string = ''
            check = None
            for i,c in enumerate(example): # char
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
                
            #print(new_string) # check the whole file
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


def find_argument_subtype(input_file,Tnum):
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
    all_subtype = data.splitlines()
    
    for line in all_subtype:
        data = line.strip().split()  
        sent_id = data[0] # Entity ID e.g., T1,E1,A1
        subtype_arg = []
        if "A" in sent_id: 
            if Tnum == data[2]:
                argument_subtype = data[3]        
    return argument_subtype


def event_argument(input_file,output_file): 
    
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
 
    # Five Type of Events 
    event_types = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
    
    all_list_arg = []
    all_list_evt = []
    all_list_sub = []

    for evtp in event_types:
        
        relations = []
        argument_out = []
        event_out  = []
        argument_sb=[]
        
        for id_event in sent_list: # Loop Events(E1,E2,...): E1 Drug
            if "E" in id_event: 
                event_type = [] 
                event_list = []
                argument_list = []
                argument_type = []
                relation = [] # [[[drug1],[argument]] [drug2,argument]]

                if dataset[id_event].split()[1].split(":")[0] == evtp:
                    event_list.append(dataset[id_event].split()[1].split(":")[1])
                    event_type.append(dataset[id_event].split()[1].split(":")[0])
                    #print('test2', event_list,event_type)

                    for i in dataset[id_event].split()[2:]:
                        argument_list.append(i.split(":")[1])# 'Status:T2', 'Type:T3', 'Type2:T3' based on E1
                        argument_type.append(i.split(":")[0]) #argument_type

                    event = attr_span(event_type,event_list,dataset)
                    argument = attr_span(argument_type,argument_list,dataset) # Event E1 Drug's arguments
                    
                    argument_in = []
                    argument_sub = []
                    
                    for key in event: # inside loop Ei key is fix
                        et = [[(event[key][1],(event[key][2],event[key][3]))]]

                    for key2 in argument:
                        arg = [(argument[key2][1],(argument[key2][2],argument[key2][3]))]
                        argument_in.append(arg)

                        subtype_T = argument[key2][0]
                        
                        if 'Status' in argument[key2][1]:
                            argument_subtype = find_argument_subtype(input_file,subtype_T)
                            argument_sub.append(argument_subtype)
                        elif 'TypeLiving' in argument[key2][1]:
                            argument_subtype = find_argument_subtype(input_file,subtype_T)
                            argument_sub.append(argument_subtype)
                        else:
                            argument_subtype = "Delete"
                            argument_sub.append(argument_subtype)

                    event_out.append(et)
                    argument_out.append(argument_in)
                    argument_sb.append(argument_sub)
                    
        all_list_evt.extend(event_out)
        all_list_arg.extend(argument_out)
        all_list_sub.extend(argument_sb)
        #print(all_list_sub)   

    return all_list_evt,all_list_arg,all_list_sub # two set event-argument 
    

def main():
    
    regex = re.compile(r'\d+')
    
    examples = [] # save all train sentences
    file_records = []
    subtype_records = []
    triggers_exs = []
    args_exs = []
    
    for filename in glob.glob('./Annotations/'+para_data['argument_train_dev']+'/mimic/*.ann'): #./Annotations/train/mimic/*.ann' 
        txt_filename = filename.replace(".ann", ".txt")
        trigger_filename = filename.replace('./Annotations/'+para_data['argument_train_dev']+'/mimic', "./Annotations/dev/test") # no use 

        #print('\ntrigger_filename:', filename)
        
        events,arguments,subtypes = event_argument(filename,trigger_filename)
        
        #for a, s in zip(arguments,subtypes):
        #    print(a,s)

        relations = [] 
        argu_subtype = []
        
        for i,c in enumerate(events):
            for j,(arg,sub) in enumerate(zip(arguments,subtypes)):
                if i == j: 
                    for l,g in zip(arguments[j],subtypes[j]):
                        relation = [] 
                        relation.append(events[i][0]) # r1
                        relation.append(l) #r2
                        relations.append(relation)  #r1+r2 
                        argu_subtype.append(g) #r3


        with open(txt_filename, 'r') as iFile: # text data
            data = iFile.read()
            
            for rl,sb in zip(relations,argu_subtype):
                #print('pair:',rl)
                triggers_ex = rl[0]   #[('Drug', (243,252))]
                args_ex = rl[1]   #[('StatusTime', (293,307))]
                example = add_tokens(data,triggers_ex,args_ex)
                idx = regex.findall(txt_filename)
                
                examples.append(example[0])
                file_records.append(idx[0])
                subtype_records.append(sb)
                triggers_exs.append(triggers_ex[0][0])
                args_exs.append(args_ex[0][0])
                  
    #list_match = ['match' for i in range(len(examples))]
    
    c = {"sentence": examples,
        #"matchorNot": list_match,
        "argument_subtype": subtype_records,
        "ann": file_records,
        "event": triggers_exs,
        "argument": args_exs}
    
    df = pd.DataFrame(c)
    
    df_all_sb = df[df['argument_subtype']!= 'Delete']
    
    options_med = ['Drug', 'Alcohol','Tobacco']
    df_med = df_all_sb[df_all_sb["event"].isin(options_med)]
    
    options_emp = ['Employment']
    df_emp = df_all_sb[df_all_sb["event"].isin(options_emp)]
    
    options_liv = ['LivingStatus']
    df_liv_Status = df_all_sb[df_all_sb["event"].isin(options_liv) & (df_all_sb["argument"] == "StatusTime")] 
    df_liv_Type = df_all_sb[df_all_sb["event"].isin(options_liv) & (df_all_sb["argument"] == "TypeLiving") ]
    
    # Exporting the DataFrame into a CSV file
    # df.to_csv('relation_match.csv', header = False) # relative position
    df_med.to_csv('subtype_'+para_data['argument_train_dev']+'_med.csv', header = False) # relative position
    df_emp.to_csv('subtype_'+para_data['argument_train_dev']+'_emp.csv', header = False) # relative position
    df_liv_Status.to_csv('subtype_'+para_data['argument_train_dev']+'_liv_status.csv', header = False) # relative position
    df_liv_Type.to_csv('subtype_'+para_data['argument_train_dev']+'_liv_type.csv', header = False) # relative position
    
    #print(df_liv_Type)
    print(len(examples))
    print(len(df_med),len(df_emp),len(df_liv_Status),len(df_liv_Type))

main()