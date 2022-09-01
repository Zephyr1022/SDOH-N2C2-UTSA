#!/usr/bin/env python3
import glob
import pandas as pd
import os, sys, re
import funztools

input_relation = sys.argv[1]
event_table_dir = sys.argv[2] # store trigger prediction ann and argument prediction 
#subtype_seed = sys.argv[3] # subtype model 
  
def arg_subtype_med(input1,input2):
    arg_subtype_med =[]
    arg_subtype =[]
    for mat, med in zip(input1.values.tolist(),input2):
        class_ = int(med[0])
        if class_ == 0:
            arg_subtype.append('current')
        elif class_ == 1:
            arg_subtype.append('none')
        else: 
            arg_subtype.append('past')

    for mat, subtype in zip(input1.values.tolist(),arg_subtype):   
        line = mat.strip().split()
        temp = line[0]+' '+line[5]+' '+ line[6]+' '+subtype
        arg_subtype_med.append(temp)
    
    return arg_subtype_med

# ['employed', 'homemaker', 'on_disability', 'retired', 'student', 'unemployed']
def arg_subtype_emp(input1,input2):
    arg_subtype_emp =[]
    arg_subtype =[]
    for mat, emp in zip(input1.values.tolist(),input2):
        class_ = int(emp[0])
        if class_ == 0:
            arg_subtype.append('employed')
        elif class_ == 1:
            arg_subtype.append('homemaker')
        elif class_ == 2:
            arg_subtype.append('on_disability')
        elif class_ == 3:
            arg_subtype.append('retired')
        elif class_ == 4:
            arg_subtype.append('student')
        else: 
            arg_subtype.append('unemployed')

    for mat, subtype in zip(input1.values.tolist(),arg_subtype):   
        line = mat.strip().split()
        temp = line[0]+' '+line[5]+' '+ line[6]+' '+subtype
        arg_subtype_emp.append(temp)
    
    return arg_subtype_emp

# ['current', 'future', 'none', 'past']
def arg_subtype_livs(input1,input2):
    arg_subtype_livs =[]
    arg_subtype =[]
    for mat, livs in zip(input1.values.tolist(),input2):
        class_ = int(livs[0])
        if class_ == 0:
            arg_subtype.append('current')
        elif class_ == 1:
            arg_subtype.append('future')
        elif class_ == 2:
            arg_subtype.append('none')
        else: 
            arg_subtype.append('past')

    for mat, subtype in zip(input1.values.tolist(),arg_subtype):   
        line = mat.strip().split()
        temp = line[0]+' '+line[5]+' '+ line[6]+' '+subtype
        arg_subtype_livs.append(temp)
    
    return arg_subtype_livs

# ['alone', 'homeless', 'with_family', 'with_others']
def arg_subtype_livt(input1,input2):
    arg_subtype_livt =[]
    arg_subtype =[]
    for mat, livt in zip(input1.values.tolist(),input2):
        class_ = int(livt[0])
        if class_ == 0:
            arg_subtype.append('alone')
        elif class_ == 1:
            arg_subtype.append('homeless')
        elif class_ == 2:
            arg_subtype.append('with_family')
        else: 
            arg_subtype.append('with_others')

    for mat, subtype in zip(input1.values.tolist(),arg_subtype):   
        line = mat.strip().split()
        temp = line[0]+' '+line[5]+' '+ line[6]+' '+subtype
        arg_subtype_livt.append(temp)
    
    return arg_subtype_livt


def fillin_events_table(input_file,number_t,argument_t):
    with open(input_file, "a") as split_trigger:
        i = number_t + 1
        #print("T",i,argument_t)
        split_trigger.write("T"+str(i)+"\t")
        split_trigger.write(' '.join(argument_t[0:3])+"\t")
        split_trigger.write(' '.join(argument_t[3:]))
        split_trigger.write('\n')
        return i

def initial_dict(input_data):
    dict_et = {}
    regex = re.compile(r'\d+')
    for a in input_data:
        EVN = a.strip().split()[0] # T1
        EVNN = a.strip().split()[1:] # ['Tobacco', '16', '26', 'Cigarettes']
        
        event_num = regex.findall(EVN)
        event_type = EVNN[0]
        dict_key = 'E'+str(event_num[0]) # E1 
        dict_value = '\t' + event_type + ':' + 'T'+ event_num[0]
        dict_et[dict_key] = [dict_value]
    #print(dict_et)
    return dict_et

def write_event(input_file,input_dict):
    dict_et = input_dict 
    with open(input_file, "a") as event_table:
        for key in dict_et:
            #print(key + ' '.join(dict_et[key]))
            event_table.write(key + ' '.join(dict_et[key]))
            event_table.write('\n')
            
def write_argument_subtype(input_file,input_list):
    #print(input_list)
    with open(input_file, "a") as event_table:
        for key in input_list:
            #print(key)
            event_table.write(key)
            event_table.write('\n')
            
def main():
    
    regex = re.compile(r'\d+')
    
    match_med = pd.read_csv('./template_rl/subtype_test22_med.csv',header = None)
    match_emp = pd.read_csv('./template_rl/subtype_test22_emp.csv',header = None)
    match_liv_status = pd.read_csv('./template_rl/subtype_test22_liv_status.csv',header = None)
    match_liv_type = pd.read_csv('./template_rl/subtype_test22_liv_type.csv',header = None)
    
    with open("./relation_pred/test22-base-pred-subtype-med-123.csv", 'r') as iFile_pred1:
        med_pred = iFile_pred1.read()   
    med_pred_ = med_pred.splitlines()
    
    with open("./relation_pred/test22-base-pred-subtype-emp-123.csv", 'r') as iFile_pred2:
        emp_pred = iFile_pred2.read()   
    emp_pred_ = emp_pred.splitlines()
    
    with open("./relation_pred/test22-base-pred-subtype-liv-status-123.csv", 'r') as iFile_pred3:
        liv_status_pred = iFile_pred3.read()   
    liv_status_pred_ = liv_status_pred.splitlines()
    
    with open("./relation_pred/test22-base-pred-subtype-liv-type-123.csv", 'r') as iFile_pred4:
        liv_type_pred = iFile_pred4.read()   
    liv_type_pred_ = liv_type_pred.splitlines()
    
    #with open("test-base-prob-V1.csv", 'r') as iFile_pred5:
    #    match_prob = iFile_pred5.read()   
    #match_prob_ = match_prob.splitlines()

    with open(input_relation,'r') as iFile_rel: #"event-argument-all-poss-relation.txt"
        #with open("event-argument-match-relation.txt", 'r') as iFile:
        eve_arg_relation = iFile_rel.read()
        
    all_relation = eve_arg_relation.splitlines() #'0477 Employment 34 44 StatusEmploy 25 31\tworked'
    
    arg_subtype_med1 = arg_subtype_med(match_med[4],med_pred_)
    arg_subtype_emp1 = arg_subtype_emp(match_emp[4],emp_pred_)
    arg_subtype_livs1 = arg_subtype_livs(match_liv_status[4],liv_status_pred_)
    arg_subtype_livt1 = arg_subtype_livt(match_liv_type[4],liv_type_pred_)
    
    all_subtype = arg_subtype_med1+arg_subtype_emp1+arg_subtype_livs1+arg_subtype_livt1
    #print(all_subtype)

    for filename in glob.glob(event_table_dir):
        with open(filename, 'r') as iFile_set:
            ann_trigger = iFile_set.read()
       # print(ann_trigger)
        
        filename2 = filename.replace('/ann', '/table') # copy new "/ann", "/table"
        print(filename2)
        
        idx = regex.findall(filename) #'0477'
        print(idx)
        ann_all = ann_trigger.splitlines()
        
        dict_et = {}
        list_subtypes = []
        sub_num = 0
        match_subtypes = []
        arg_dup = {}
        
        argument_contents = set()
        argument_subT = set()
       
        if len(ann_all) != 0:
            last_t = regex.findall(ann_all[-1].strip().split()[0])
            last_t = int(last_t[0])
            #print(last_t)
            
        dict_et = initial_dict(ann_all)
        #print("initial:", dict_et)
        
        for sb in all_subtype:
            sbb = sb.strip().split()
            if sbb[0] == idx[1]:
                match_subtypes.append(sbb[0]+' '+sbb[1]+' '+sbb[2]+' '+sbb[3])           
        #print("match_subtypes",match_subtypes) #subtype
        
        for a in ann_all: # trigger ann           
            EVN = a.strip().split()[0] # T1
            EVNN = a.strip().split()[1:] # ['Tobacco', '16', '26', 'Cigarettes']
            event_num = regex.findall(EVN) #1
            dict_key = 'E'+str(event_num[0]) # E1
            #print(type(EVN))
            #print('trigger:', a.strip().split())
            #break

            for argument_line in all_relation: # 0446 Drug 70 83 Type 64 77    Crack Cocaine
                #print(argument_line,argument_line.strip().split()[0],idx[0])
                #print(argument_line.strip().split()[0],idx[0])
                if argument_line.strip().split()[0] == idx[1]:
                    event_type = argument_line.strip().split()[1] # relation trigger 
                    event_start = argument_line.strip().split()[2] # trigger start 
                    event_end = argument_line.strip().split()[3] # trigger end
                    argument_type = argument_line.strip().split()[4] # argument
                    argument_start = argument_line.strip().split()[5] # argument                     
                    argument_end = argument_line.strip().split()[6] # argument 
                    argument_content = argument_line.strip().split()[4:]
                    
                    #print("filenumber",argument_start,argument_end)
                    #print(argument_content,"test",event_type,event_start,event_end,EVNN[0],EVNN[1],EVNN[2])
                    
                    count_status = 0
                    #print(event_type,EVNN[0],event_start,EVNN[1],event_end,EVNN[2])
                    if event_type == EVNN[0] and event_start == EVNN[1] and event_end == EVNN[2]:
                        #write argument Ti in new ann file
                        #if not argument_content in argument_contents:
                        #print(''.join(argument_content))
                        #print("all-poss-relation:", event_type,event_start,event_end,"EVANN",EVNN[0],EVNN[1],EVNN[2])
                        #print('argument_content:',argument_content)                        
                        if ''.join(argument_content) not in argument_contents:
                            argument_contents.add(''.join(argument_content))                           
                            argument_num = fillin_events_table(filename2,last_t,argument_content)
                            arg_dup[''.join(argument_content)] = argument_num
                            #print(argument_contents)
                            #break
                        else: # 不加 argument 
                            argument_num = arg_dup[''.join(argument_content)]
                            
                        # prepare Event list 
                        if "Status" in argument_type:
                            argument_type = "Status"
                            for match_subtype in match_subtypes:
                                if match_subtype.strip().split()[1] == argument_content[1] and match_subtype.strip().split()[2] == argument_content[2]:  
                                    if not ''.join('TypeLivingVal'+' ' +'T'+ str(argument_num)) in argument_subT:
                                        sub_num += 1
                                        argument_subT.add(''.join('TypeLivingVal'+' ' +'T'+ str(argument_num)))
                                        
                                        if argument_content[0] == 'StatusEmploy':
                                            list_subtype = 'A'+ str(sub_num) +'\t'+ 'StatusEmployVal'+' ' +'T'+ str(argument_num) +' '+match_subtype.strip().split()[3]
                                    
                                        else:                     
                                            list_subtype = 'A'+ str(sub_num) +'\t'+ 'StatusTimeVal'+' ' +'T'+ str(argument_num) +' '+match_subtype.strip().split()[3]                                       
                                        list_subtypes.append(list_subtype)
                         
                            # A2    StatusTimeVal T3 current
                            # StatusEmployVal, StatusTimeVal,TypeLivingVal 
                        if "TypeLiving" in argument_type:
                            argument_type = "Type"
                            for match_subtype in match_subtypes:
                                if match_subtype.strip().split()[1] == argument_content[1] and match_subtype.strip().split()[2] == argument_content[2]:  
                                    if not ''.join('TypeLivingVal'+' ' +'T'+ str(argument_num)) in argument_subT:
                                        sub_num += 1
                                        argument_subT.add(''.join('TypeLivingVal'+' ' +'T'+ str(argument_num)))
                                        
                                        list_subtype = 'A'+ str(sub_num) +'\t'+ 'TypeLivingVal'+' ' +'T'+ str(argument_num) +' '+match_subtype.strip().split()[3]
                                        list_subtypes.append(list_subtype)
                        
                        dict_value = argument_type+':'+'T'+ str(argument_num)
                        #print(dict_value)                        
                        dict_et[dict_key].append(dict_value)
                        last_t += 1
                        
                        #print('E', event_num[0], '\t', event_type, ':' , 'T', event_num[0],argument_type,':', 'T', argument_num)
        #print(list_subtypes)
        write_event(filename2,dict_et)
        write_argument_subtype(filename2,list_subtypes)
main()
