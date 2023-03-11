#!/usr/bin/env python3

# python 读取文件，将文件中空白行去掉

import random
import glob
from sklearn.model_selection import train_test_split
import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

#input_text = sys.argv[1] #search_data1.yaml
output_text = sys.argv[1]
#para_data = read_data.ReadData(input_text).loading_data() # ner_conll = sys.argv[2]

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
				
# shuffle train				
def shuffle_conll_data(input_file, output_file, random_seed=42):
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
	for text in data.split("\n\n"):
		sent_id = text.splitlines()[0].strip().split()[-1]
		dataset[sent_id] = text
		
	with open(output_file, "w") as train_shuffle:
		for i in ids: # Save data based on shuffled ids
			train_shuffle.write(dataset[i])
			if i != ids[-1]:
				train_shuffle.write('\n\n')       

# regenerete the conll file as input				
def del_num(input_file,output_file): 
	with open(input_file, 'r') as iFile:
		data = iFile.read()
	with open(output_file, "w") as train_trigger:  
		for line in data.splitlines():
			data = line.strip().split()
			if len(data) != 0:
				train_trigger.write(' '.join(data[:3]))
				train_trigger.write("\n")
			else:
				train_trigger.write("\n")
				
def main():
	regex = re.compile(r'\d+')
	with open("sdoh.conll", "w") as train_trigger:
		for filename in glob.glob('./Annotations/triggers/train/*.conll'):
			conll_filename = regex.findall(filename)
			with open(filename, 'r') as iFile:
				data = iFile.read()
			all_data = data.splitlines()
			for line in all_data:
				if line != "":
					data = line.strip().split()  
					train_trigger.write(data[3]+" "+ "N"+" "+  data[0]+" "+ data[1]+" "+ data[2]+" "+ conll_filename[0])
					train_trigger.write("\n")
				else:
					train_trigger.write("\n")	
			train_trigger.write("\n")
	
	# generate sdoh.conll
	delblankline("sdoh.conll","temp.conll")  
	clean_file("temp.conll","temp2.conll")
	
	#split_conll_data("temp2.conll",'train_sdoh.conll','dev_sdoh.conll',123)
	#spaceline("train_sdoh.conll","temp3.conll")
	#spaceline("dev_sdoh.conll","dev_sdoh_temp.conll")
	
	spaceline("temp2.conll","temp3.conll")
	shuffle_conll_data("temp3.conll", 'sdoh_shuffle.conll',123)
	del_num("sdoh_shuffle.conll", output_text) # 'train_trigger_ner.txt')
	#del_num("dev_sdoh_temp.conll", 'dev_sdoh.txt')
	print("triggers ner train done")
	
main()
