import glob
import os
import sys

def main():
    directory = sys.argv[1]
    argument = sys.argv[2]
    output_filename= sys.argv[3]
    print(sys.argv)
    outputFile = open(output_filename, 'w')
    for j,filename in enumerate(glob.glob(directory+'*.conll')):
        if j > 0:
            outputFile.write(os.linesep+os.linesep)
        with open(filename) as iFile:
            data = iFile.read()
        items = data.split(os.linesep + os.linesep)
        total = len(items)
        for i,x in enumerate(items):
            new_item = 'O\t0\t0\t<{}>'.format(argument)
            new_item = new_item+os.linesep+x
            tmp = new_item.splitlines()
            tmp_total = len(tmp)
            for k,line in enumerate(tmp):
                outputFile.write(line+"\t"+filename)
                #new_item += os.linesep + os.linesep
                if k+1 < tmp_total:
                    outputFile.write(os.linesep)
            if i+1 < total:
                outputFile.write(os.linesep+os.linesep)
    outputFile.close()
main()