#!/bin/bash

# trigger ner model training
# option 1-tag trigger
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-event-tag-uw.yaml > ./ner_results/trigger_tag_ner_train_uw.out 2>&1 &
# option 2-notag trigger
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-26-event-uw.yaml > ./ner_results/trigger_ner_train_uw.out 2>&1 &




# argument ner model training
# option 1 together:
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer.py sdoh-26-arg-together-uw.yaml > ./ner_results/train_arg_together_uw.out 2>&1 &
# option 2 separate:
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-drug-uw.yaml > ./ner_results/train_ner_drug_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-alcohol-uw.yaml > ./ner_results/train_ner_alcohol.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-tobacco-uw.yaml > ./ner_results/train_ner_tobacco.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-26-livingstatus-uw.yaml > ./ner_results/train_ner_liv_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-employment-uw.yaml > ./ner_results/train_ner_emp_overlap.out 2>&1 &



# Evaluation
# model prediction - using trained model to predict - get test_argu_drug_pred.txt
CUDA_VISIBLE_DEVICES=0 python error_analysis.py sdoh-26-drug-uw.yaml > ./ner_results/pred_ner_drug_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python error_analysis.py sdoh-26-alcohol-uw.yaml > ./ner_results/pred_ner_alcohol_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python error_analysis.py sdoh-26-tobacco-uw.yaml > ./ner_results/pred_ner_tobacco_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 python error_analysis.py sdoh-26-livingstatus-uw.yaml > ./ner_results/pred_ner_liv_uw.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python error_analysis.py sdoh-26-employment-uw.yaml > ./ner_results/pred_ner_emp_uw.out 2>&1 &	


