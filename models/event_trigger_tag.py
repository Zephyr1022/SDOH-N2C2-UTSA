#!/usr/bin/env python3

#!/usr/bin/env python3

import glob
import os, sys
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

input_trigger = sys.argv[1]
input_dir = sys.argv[2]
train_dev_test = sys.argv[3]  # output

#input_text = sys.argv[1]
#para_data = read_data.ReadData(input_text).loading_data()

def extract_trigger(input_file,output_file): 
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
    sent_list = []
    event_list = []
    trigger = "E"
    dataset = {} # Put dataset into dictionary where key is the id
    
    for line in data.splitlines():
        data = line.strip().split()  
        sent_id = data[0] # Entity ID e.g., T1
#        print(sent_id)
        dataset[sent_id] = line.strip() #data[1:] # Stores entity info
    
        if trigger in sent_id:
#           print(data[1].split(":")[1])
            event_list.append(data[1].split(":")[1])
            
    with open(output_file, "w") as train_trigger:
        for id in event_list:
            if dataset[id].strip().split()[1] == input_trigger: # only save Drug eneities 
                train_trigger.write(dataset[id]+'\n')

def main():
    
    for filename in glob.glob(input_dir + '/*.ann'): 
        
        trigger_filename = filename.replace(input_dir, './Annotations/triggers_tag/'+ input_trigger + train_dev_test) #'/test'
        extract_trigger(filename,trigger_filename) 
        
        #~/sdoh/Annotations/test/test3
        #trigger_filename = filename.replace("Annotations/dev/mimic", "Annotations/val")
        # './Annotations/events_tag/'+ trigger +'/test'
main()

        
