import glob
import os
import sys

def main():
    directory = sys.argv[1]
    argument = sys.argv[2]
    output_filename= sys.argv[3]
    print(sys.argv)
    
    outputFile = open(output_filename, 'w')
#   for j,filename in enumerate(glob.glob(directory+'*.conll')):
#       if j > 0:
#           outputFile.write(os.linesep+os.linesep)
    with open(directory) as iFile:
        data = iFile.read()
    items = data.split(os.linesep + os.linesep)
    total = len(items)
    
    for i,x in enumerate(items):
        # Trigger
        if argument == "Trigger":
            trigger_type = x.split(os.linesep)[0].split(' ')[0]
#           trigger_line = x.split(os.linesep)[0]
            new_item = x.split(os.linesep)[0].replace(trigger_type, '<Trigger>')
#           content = x.replace(trigger_type+' '+'N O\n', '')
#           content = x.replace(trigger_line+'\n', '')
#           print(content)
#           break
            
        # Argument
        if argument == "Argument":
            argument_type = x.split(os.linesep)[0].split(' ')[0]
            new_item = x.split(os.linesep)[0].replace(argument_type, '<Argument>')
        
#       new_item = '<{}> N O'.format(argument)
        new_item = new_item+os.linesep+x
#       print(trigger_line)
#       print(content)
#       new_item = trigger_line+os.linesep+new_item+os.linesep+content
#       print(new_item)
        
        tmp = new_item.splitlines()
        tmp_total = len(tmp)
        for k,line in enumerate(tmp):
            # outputFile.write(line+"\t"+filename)
            outputFile.write(line)
            #new_item += os.linesep + os.linesep
            if k+1 < tmp_total:
                outputFile.write(os.linesep)
        if i+1 < total:
            outputFile.write(os.linesep+os.linesep)
            
    outputFile.close()
main()