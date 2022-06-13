#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

# python 读取文件，将文件中空白行去掉

import random
import glob
from sklearn.model_selection import train_test_split
import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

test_dir = sys.argv[1]
output_ner = sys.argv[2]

# step 1 save based on the trigger: drug,alcohol,tabacoo, emp,liv
# step 2 save based on train, dev, test
# step 3 exhausted search 5 type of argument: and combine conll files 
# For example, NER/Drug/Train/ including drug-5-related-argu-combination
# para_data['event_type'] : Drug    para_data['argument_train_dev'] : train: ./NER/Drug/train/*.conll

def combine_conll(output_file):
    
    regex = re.compile(r'\d+')
    
    with open(output_file,"w") as train_trigger: #sdoh.conll
        for filename in glob.glob( test_dir +'*.conll'):
            with open(filename, 'r') as iFile:
                data = iFile.read()

            all_data = data.splitlines()
            #print(all_data)
            for line in all_data:
                if line != "":
                    data = line.strip().split()  
                    idx = regex.findall(data[4])
                    #print(idx)
                    train_trigger.write(data[3]+" "+ "N"+" "+  data[0]+" "+ data[1]+" "+ data[2]+" "+ idx[0])
                    train_trigger.write("\n")
                else:
                    train_trigger.write("\n")
            train_trigger.write("\n")
            
def delblankline(input_file,output_file):
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    with open(output_file, "w") as train_del:    
        for line in data.split("\n\n"):
            train_del.write(line)
            train_del.write("\n")
                     
# function to check if the last line can be deleted
def clean_file(input_file,output_file):
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    with open(output_file, "w") as train_clean:
        for line in data.splitlines(): #lline.strip()
            if line != "":
                train_clean.write(line)
                train_clean.write("\n")
                                
def split_conll_data(input_file, output_file1,output_file2, random_seed=42):
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    ids = set()
    for line in data.splitlines(): # Get all IDs
        if len(line.strip().split()) == 0:
            continue
        sent_id = line.strip().split()[-1]
        ids.add(sent_id)
    
    ids = sorted(list(ids)) # Sort ids
    random.seed(random_seed)
    random.shuffle(ids) # Shuffle ids
    
    list_train, list_dev = train_test_split(ids, train_size = 0.8,random_state = random_seed)
    #print(len(list_train),len(list_dev))
    
    with open(output_file1, "w") as train_split:  
        for line in data.splitlines():
            if len(line.strip().split()) == 0:
                continue
            
            elif line.strip().split()[5] in list_train:
                train_split.write(' '.join(line.strip().split()[:6]))
                train_split.write("\n")
                
    with open(output_file2, "w") as dev_split:  
        
        for line in data.splitlines():
            if len(line.strip().split()) == 0:
                continue
            
            elif line.strip().split()[5] in list_dev:
                dev_split.write(' '.join(line.strip().split()[:6]))
                dev_split.write("\n")
                
# add space b/w diff files
def spaceline(input_file,output_file):
    with open(input_file, 'r') as iFile:
        data = iFile.read()
    with open(output_file, "w") as train_space:
        prev = None
        for line in data.splitlines(): # Get all IDs
            sent_id = line.strip().split()[-1]    
            if prev == None:
                prev = line.strip().split()[-1]
                train_space.write(line)
                train_space.write("\n")
            elif sent_id != prev:
                train_space.write("\n")
                prev = sent_id #update
                train_space.write(line)
                train_space.write("\n")
            else:
                train_space.write(line)
                train_space.write("\n")
                
# shuffle train
def shuffle_conll_data(input_file, output_file, random_seed=123):
    with open(input_file, 'r') as iFile:
        data = iFile.read()
    ids = set()
    for line in data.splitlines(): # Get all IDs
        if len(line.strip().split()) == 0:
            continue
        sent_id = line.strip().split()[-1]
        ids.add(sent_id)
    
    ids = sorted(list(ids)) # Sort ids 
    random.seed(random_seed)
    random.shuffle(ids) # Shuffle ids
    dataset = {} # Put dataset into dictionary where key is the id
    for key_ids in ids:
        dataset[key_ids]=[]
    for text in data.split("\n\n"):  
        sent_id = text.splitlines()[0].strip().split()[-1]    
        dataset[sent_id].append(text)
    #print(' '.join(dataset['4883']))
        
    with open(output_file, "w") as train_shuffle:
        for i in ids: # Save data based on shuffled ids
            train_shuffle.write('\n'.join(dataset[str(i)]))
            if i != ids[-1]:
                train_shuffle.write('\n\n') 
                
def spaceline_tag(input_file,output_file):
    trigger_types = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
    with open(input_file, 'r') as iFile:
        data = iFile.read()
    with open(output_file, "w") as train_space:
        all_data = data.splitlines()
        size = len(all_data)
        for l,line in enumerate(all_data):
            if l == 0:
                train_space.write(line)
                train_space.write("\n")
            if l > 0 and l < size-1 and line != "":
                trigger_tag = line.strip().split()[0].replace('<','').replace('>','')
                sent_id = line.strip().split()[-1]
                start = int(line.strip().split()[3])
                end = int(line.strip().split()[4])  
                if start == 0 and end == 0 and trigger_tag in trigger_types: 
                    #print("found it")
                    train_space.write("\n")
                    train_space.write(line)
                    train_space.write("\n")
                else:
                    train_space.write(line)
                    train_space.write("\n")  
            if l == size-1:
                train_space.write(line)
                
# regenerete the conll file as input                
def del_num(input_file,output_file): 
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
    with open(output_file, "w") as train_trigger:
        all_data = data.splitlines() # call once not in loop
        for l,line in enumerate(all_data): 
            ner_data = line.strip().split()
            if l < len(all_data)-1:
                if len(ner_data) != 0: # 非空行
                    train_trigger.write(' '.join(ner_data[:3]))
                    train_trigger.write("\n")
                else:
                    train_trigger.write("\n")   
            else:
                train_trigger.write(' '.join(ner_data[:3]))
                
def main():
    
   # print("test")
    #print("STEP 1")
    combine_conll('sdoh_temp.conll')# 'sdoh_'+para_data['argument_file']+'-train.conll'
    #print("STEP 2")
    delblankline('sdoh_temp.conll',"temp.conll") # 'sdoh_'+para_data['argument_file']+'-train.conll'
    #print("STEP 3")
    clean_file("temp.conll","temp2.conll")
    #print("STEP 4")
    spaceline("temp2.conll","temp3.conll")
    #print("STEP 5")
    shuffle_conll_data("temp3.conll", 'sdoh_shuffle.conll',123)
    #print("STEP 6")
    spaceline_tag("sdoh_shuffle.conll","tag.conll") # 没问题
    #print("STEP 7")
    del_num("tag.conll", output_ner) #出现问题 # train_argu_drug_ner.txt
    print("train done")
    
main()
