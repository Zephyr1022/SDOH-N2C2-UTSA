#!/bin/bash

# Evaluation

# trigger 

# Option 1 with tag 
CUDA_VISIBLE_DEVICES=1 nohup python error_analysis.py sdoh-26-event-tag.yaml > ./ner_results/trigger_tag_ner_pred.out 2>&1 &

#Option 2-1 no tag 
CUDA_VISIBLE_DEVICES=1 nohup python error_analysis.py sdoh-26-event.yaml > ./ner_results/trigger_ner_pred26.out 2>&1 &

#Option 2-2 no tag with bert 
CUDA_VISIBLE_DEVICES=1 nohup python error_analysis.py sdoh-84-event.yaml > ./ner_results/trigger_ner_pred84.out 2>&1 &

#Option 2-3 no tag with word 
CUDA_VISIBLE_DEVICES=2 nohup python error_analysis.py sdoh-40-event.yaml > ./ner_results/trigger_ner_pred40.out 2>&1 &

#Option 1 with tag 
python pred2ann_events.py sdoh-26-event-tag.yaml './test_pred/test_events_tag_pred.txt' 'events_tag_relation_pred.txt'

#Option 2-1 no tag
python pred2ann_events.py sdoh-26-event.yaml './test_pred/test_events_pred26.txt' 'events26_relation_pred.txt' 

#Option 2-2 no tag
python pred2ann_events.py sdoh-84-event.yaml './test_pred/test_events_pred84.txt' 'events84_relation_pred.txt'  

#Option 2-3 no tag
python pred2ann_events.py sdoh-40-event.yaml './test_pred/test_events_pred40.txt' 'events40_relation_pred.txt'


# argument 

# option 1 together 
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-26-only-arg-new.yaml > ./ner_results/train_only_arg_new.out 2>&1 &


# option 1 seperate
CUDA_VISIBLE_DEVICES=0 python error_analysis.py sdoh-26-drug-uw.yaml > ./ner_results/pred_ner_drug_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python error_analysis.py sdoh-26-alcohol-uw.yaml > ./ner_results/pred_ner_alcohol_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python error_analysis.py sdoh-26-tobacco-uw.yaml > ./ner_results/pred_ner_tobacco_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 python error_analysis.py sdoh-26-livingstatus-uw.yaml > ./ner_results/pred_ner_liv_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python error_analysis.py sdoh-26-employment-uw.yaml > ./ner_results/pred_ner_emp_uw.out 2>&1 &	


