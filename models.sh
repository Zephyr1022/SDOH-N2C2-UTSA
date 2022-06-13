#!/bin/bash

conda activate scispacyV5

# NER 

# trigger ner model training

# option 1-tag trigger
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-event-tag-uw.yaml > ./ner_results/trigger_tag_ner_train_uw.out 2>&1 &

# option 2-notag trigger  with flair
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-26-event-uw.yaml > ./ner_results/trigger_ner_train_uw.out 2>&1 &
#Option 2-2 no tag with bert 
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-84-event.yaml > ./ner_results/trigger_ner_train_84.out 2>&1 &
#Option 2-3 no tag with word 
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-40-event.yaml > ./ner_results/trigger_ner_train_40.out 2>&1 &


# argument ner model training

# option 1 together:
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer.py sdoh-26-arg-together-uw.yaml > ./ner_results/train_arg_together_uw.out 2>&1 &

# option 2 separate:
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-drug-uw.yaml > ./ner_results/train_ner_drug_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-alcohol-uw.yaml > ./ner_results/train_ner_alcohol.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-tobacco-uw.yaml > ./ner_results/train_ner_tobacco.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-26-livingstatus-uw.yaml > ./ner_results/train_ner_liv_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-employment-uw.yaml > ./ner_results/train_ner_emp_overlap.out 2>&1 &


