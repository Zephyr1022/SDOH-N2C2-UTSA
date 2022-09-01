#!/bin/bash argumentr_single_overlap_test.py

#!/usr/bin/env python3
# ALL

for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
    
do 
    echo "$argument_type"
    
    # generate all ann files for 'Status' in diff folder
    python argument_extract_all_test.py sdoh-26-drug.yaml $argument_type test > overlap_results/test_drug_$argument_type.out 2>&1
    
    python argument_extract_all_test.py sdoh-26-alcohol.yaml $argument_type test > overlap_results/test_alcohol_$argument_type.out 2>&1
    
    python argument_extract_all_test.py sdoh-26-tobacco.yaml $argument_type test > overlap_results/test_tobacco_$argument_type.out 2>&1
   
    # convert ann&txt to conll, 6 produces
    bash arguments_med_test.sh > anntoconll_results/arguments_med_test.out 2>&1
    
    
    # combine conll files and add tag
    python process.py ./Annotations/argu_drug/test/ $argument_type test_drug_$argument_type.conll
    
    python process.py ./Annotations/argu_alcohol/test/ $argument_type test_alcohol_$argument_type.conll
    
    python process.py ./Annotations/argu_tobacco/test/ $argument_type test_tobacco_$argument_type.conll
    
    
    # remove ann and conll in the folder and ready for next argument_type
    bash rm_med_test.sh
    
    
done

echo "test done1"


for argument_type_ in 'Status' 'Duration' 'History' 'Type'
do 
    echo "$argument_type_"
    python argument_extract_all_test.py sdoh-26-livingstatus.yaml $argument_type_ test > overlap_results/test_livingstatus_$argument_type_.out 2>&1
    
    python argument_extract_all_test.py sdoh-26-employment.yaml $argument_type_ test > overlap_results/test_employment_$argument_type_.out 2>&1
    
    # convert ann&txt to conll, 4 procedues
    bash arguments_env_test.sh > anntoconll_results/arguments_env_test.out 2>&1
    
    # combine conll files and add tag
    python process.py ./Annotations/argu_liv/test/ $argument_type_ test_livingstatus_$argument_type_.conll
    
    python process.py ./Annotations/argu_emp/test/ $argument_type_ test_employment_$argument_type_.conll
    
    # remove ann and conll in the folder and ready for next argument_type
    bash rm_env_test.sh
    
    
done

echo "done2"
