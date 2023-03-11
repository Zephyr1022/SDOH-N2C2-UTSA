#!/usr/bin/env python3

#!/usr/bin/env python3

import glob
import os, sys
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

input_text = sys.argv[1] #search_data1.yaml
train_dev_test = sys.argv[2]
config = read_data.ReadData(input_text).loading_data()

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
#            print(data[1].split(":")[1])
			event_list.append(data[1].split(":")[1])
			
	with open(output_file, "w") as train_trigger:
		for id in event_list:
			'''
			train_trigger.write(str(id)+"\t")
			train_trigger.write(dataset[id][0]+" "+ dataset[id][1]+ " "+ dataset[id][2]+'\t')
			train_trigger.write(' '.join(dataset[id][3:]))
			'''
			train_trigger.write(dataset[id]+'\n')
			
def main():
	
	for filename in glob.glob('./Annotations/'+ train_dev_test +'/mimic/*.ann'):
		trigger_filename = filename.replace('/Annotations/'+ train_dev_test  +'/mimic', '/Annotations/'+ config["argument_file"] +'/'+ train_dev_test)
		
		#trigger_filename = filename.replace("Annotations/dev/mimic", "Annotations/val")
		extract_trigger(filename,trigger_filename)
		
		# Write code to read file line-by-line
		# Wrte code to save a new .ann file only containing triggers (new_data directory)
		
main()
