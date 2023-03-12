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

input_text = sys.argv[1] #search_data1.yaml
output_text = sys.argv[2] #search_data1.yaml
para_data = read_data.ReadData(input_text).loading_data()

def combine_conll(output_file):
    regex = re.compile(r'\d+')
    with open(output_file,"w") as train_trigger: #sdoh.conll
        for filename in glob.glob('./NER/'+ para_data['event_type'] + '/dev/*.conll'):
            with open(filename, 'r') as iFile:
                data = iFile.read()
                
            for line in data.splitlines():
                if line != "":
                    data = line.strip().split()  
                    idx = regex.findall(data[4])
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

# add spave b/w diff files    
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

# sort ids and put same conll with same filename together 
def sort_conll_data(input_file, output_file):
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    ids = set()
    for line in data.splitlines(): # Get all IDs
        if len(line.strip().split()) == 0:
            continue
        sent_id = line.strip().split()[-1]
        ids.add(sent_id)
    
    ids = sorted(list(ids)) # Sort ids 
    #random.seed(random_seed)
    #random.shuffle(ids) # Shuffle ids
    
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
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
#    print("Tag Part 1")
    with open(output_file, "w") as train_space:
        all_data = data.splitlines()
        size = len(all_data)
        for l,line in enumerate(all_data):
            if l == 0:
                train_space.write(line)
                train_space.write("\n")
                #print(line)
                
            if l > 0 and l < size-1 and line != "":
                sent_id = line.strip().split()[-1]
                start = int(line.strip().split()[3])
                end = int(line.strip().split()[4])  
                if start == 0 and end == 0: 
                    #print("found it")
                    train_space.write("\n")
                    train_space.write(line)
                    train_space.write("\n")
                else:
                    train_space.write(line)
                    train_space.write("\n")
                    
            if l == size-1:
                train_space.write(line)
                #print("last line:", line)
                
                
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
    
    combine_conll('sdoh_temp.conll') #'+para_data['argument_file']+'-dev
    
    # generate sdoh.conll
    delblankline('sdoh_temp.conll',"temp.conll")   # '+para_data['argument_file']+'-dev
    clean_file("temp.conll","temp2.conll")
    spaceline("temp2.conll","temp3.conll")
    sort_conll_data("temp3.conll", 'sdoh_shuffle.conll')

    spaceline_tag("sdoh_shuffle.conll",output_text + '.conll') # 'tag_'+para_data['argument_file']+'_dev.conll'
    del_num(output_text + '.conll', output_text +'_ner.txt')
    
    print("dev test done")
    
main()
