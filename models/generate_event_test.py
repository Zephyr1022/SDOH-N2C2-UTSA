#!/usr/bin/env python3

# python 读取文件，将文件中空白行去掉
import os, sys,re
import glob
import funztools
import random
from sklearn.model_selection import train_test_split

test_dir = sys.argv[1]
output_ner = sys.argv[2]

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
        for filename in glob.glob( test_dir + '*.conll'):  # ./Annotations/events/dev/
            conll_filename = regex.findall(filename)
            #print(conll_filename[1])
            
            with open(filename, 'r') as iFile:
                data = iFile.read()
                
            all_data = data.splitlines()
            #print(all_data)
            for line in all_data:
                if line != "":
                    data = line.strip().split()
                    #print(data)
                    train_trigger.write(data[3]+" "+ "N"+" "+  data[0]+" "+ data[1]+" "+ data[2]+" "+ conll_filename[0])
                    train_trigger.write("\n")
                else:
                    train_trigger.write("\n")    
            train_trigger.write("\n")
            
    # generate sdoh.conll
    delblankline("sdoh.conll","temp.conll")  
    clean_file("temp.conll","temp2.conll")
    
    spaceline("temp2.conll",output_ner.replace('ner.txt','num.conll'))
    del_num(output_ner.replace('ner.txt','num.conll'),output_ner)
    
    print("trigger ner test done")
    
main()
