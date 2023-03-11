import random
import glob
from sklearn.model_selection import train_test_split
import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

trigger_tag = sys.argv[1]
argument_tag = sys.argv[2]
input_file = sys.argv[3]
output_file = sys.argv[4]

def spaceline_tag(input_file,output_file):
    regex = re.compile(r'\d+')
    
    with open(input_file, 'r') as iFile:
        data = iFile.read()
        
#    print("Tag Part 1")
    with open(output_file, "w") as train_space:
        all_data = data.splitlines()
        size = len(all_data)
        for l,line in enumerate(all_data):
            if l == 0:
                train_space.write(line.replace(argument_tag, trigger_tag))
                train_space.write("\n")
                train_space.write(line)
                train_space.write("\n")
                #print(line)
                
            if l > 0 and l < size-1 and line != "":
                idx = regex.findall(line.strip().split()[-1])
                #print(idx)
                sent_id = idx[0] #line.strip().split()[-1]
                start = int(line.strip().split()[1])
                end = int(line.strip().split()[2])  
                if start == 0 and end == 0: 
                    #print("found it")
                    train_space.write("\n")
                    train_space.write(line.replace(argument_tag, trigger_tag))
                    train_space.write("\n")
                    train_space.write(line)
                    train_space.write("\n")
                else:
                    train_space.write(line)
                    train_space.write("\n")
                    
            if l == size-1:
                train_space.write(line)
                #print("last line:", line)
                
def main():
    
    spaceline_tag(input_file,output_file) # 没问题
    
main()