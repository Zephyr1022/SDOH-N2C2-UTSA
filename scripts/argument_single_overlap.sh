#!/bin/bash

#!/usr/bin/env python3
# ALL

for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
    
do 
    echo "$argument_type"
    
    # generate all ann files for 'Status' in diff folder
    python argument_extract_all.py sdoh-26-drug.yaml train $argument_type > ./overlap_results/train_drug_$argument_type.out 2>&1
    python argument_extract_all.py sdoh-26-drug.yaml dev $argument_type > ./overlap_results/dev_drug_$argument_type.out 2>&1
    
    python argument_extract_all.py sdoh-26-alcohol.yaml train $argument_type > ./overlap_results/train_alcohol_$argument_type.out 2>&1
    python argument_extract_all.py sdoh-26-alcohol.yaml dev $argument_type > ./overlap_results/dev_alcohol_$argument_type.out 2>&1
    
    python argument_extract_all.py sdoh-26-tobacco.yaml train $argument_type > ./overlap_results/train_tobacco_$argument_type.out 2>&1
    python argument_extract_all.py sdoh-26-tobacco.yaml dev $argument_type > ./overlap_results/dev_tobacco_$argument_type.out 2>&1

    
    # convert ann&txt to conll, 6 produces
    bash arguments_med.sh > ./anntoconll_results/arguments_med.out 2>&1
    
    
    # combine conll files and add tag
    python process.py ./Annotations/argu_drug/train/ $argument_type train_drug_$argument_type.conll
    python process.py ./Annotations/argu_drug/dev/ $argument_type dev_drug_$argument_type.conll
    
    python process.py ./Annotations/argu_alcohol/train/ $argument_type train_alcohol_$argument_type.conll
    python process.py ./Annotations/argu_alcohol/dev/ $argument_type dev_alcohol_$argument_type.conll
    
    python process.py ./Annotations/argu_tobacco/train/ $argument_type train_tobacco_$argument_type.conll
    python process.py ./Annotations/argu_tobacco/dev/ $argument_type dev_tobacco_$argument_type.conll
    
    
    # remove ann and conll in the folder and ready for next argument_type
    bash rm_med.sh
    
    
done

echo "done1"


for argument_type_ in 'Status' 'Duration' 'History' 'Type'
do 
    echo "$argument_type_"
    python argument_extract_all.py sdoh-26-livingstatus.yaml train $argument_type_ > ./overlap_results/train_livingstatus_$argument_type_.out 2>&1
    python argument_extract_all.py sdoh-26-livingstatus.yaml dev $argument_type_ > ./overlap_results/dev_livingstatus_$argument_type_.out 2>&1
    
    python argument_extract_all.py sdoh-26-employment.yaml train $argument_type_ > ./overlap_results/train_employment_$argument_type_.out 2>&1
    python argument_extract_all.py sdoh-26-employment.yaml dev $argument_type_ > ./overlap_results/dev_employment_$argument_type_.out 2>&1
    
    # convert ann&txt to conll, 4 procedues
    bash arguments_env.sh > ./anntoconll_results/arguments_env.out 2>&1
    
    # combine conll files and add tag
    python process.py ./Annotations/argu_liv/train/ $argument_type_ train_livingstatus_$argument_type_.conll
    python process.py ./Annotations/argu_liv/dev/ $argument_type_ dev_livingstatus_$argument_type_.conll
    
    python process.py ./Annotations/argu_emp/train/ $argument_type_ train_employment_$argument_type_.conll
    python process.py ./Annotations/argu_emp/dev/ $argument_type_ dev_employment_$argument_type_.conll
    
    # remove ann and conll in the folder and ready for next argument_type
    bash rm_env.sh
    
    
done

echo "done2"
