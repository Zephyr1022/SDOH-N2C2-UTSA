import os, sys,re
import glob
import funztools
from funztools.yamlbase import read_data 
from funztools.tools import score_split #file name

#input_text = sys.argv[1] #search_data1.yaml
#test_dir = sys.argv[1]
#para_data = read_data.ReadData(input_text).loading_data()

def main():
    for filename in glob.glob('./Annotations/test/*.txt'): 
        #./Annotations/train/mimic/*.ann' ~/sdoh/Annotations/test/test1
        
        ann_filename = filename.replace('txt', 'ann') 
        
        with open(ann_filename, "w") as ann_empty:
            ann_empty.write('')

main()
