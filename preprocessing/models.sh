#!/bin/bash

conda activate scispacyV5

# NER save in ./taggers 

# trigger ner model training

#option 1-tag trigger
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-event-tag-uw.yaml > ./ner_results/trigger_tag_ner_train_uw.out 2>&1 &

# clinical Bert
# CUDA_VISIBLE_DEVICES=0 python sdoh_trainer.py sdoh-26-event-tag-uw.yaml 

#option 2-1 notag trigger  with flair
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-26-event-uw.yaml > ./ner_results/trigger_ner_train_uw.out 2>&1 &
#option 2-2 no tag with bert 
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-84-event.yaml > ./ner_results/trigger_ner_train_84.out 2>&1 &
#option 2-3 no tag with word 
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-40-event.yaml > ./ner_results/trigger_ner_train_40.out 2>&1 &


# Task C

# Clinical Bert trigger_tag + bert
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-84-event.yaml > ./ner_results/trigger_tag_ner_train_uw_clinical.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-84-event.yaml > ./ner_results/trigger_tag_ner_train_uw-bert.out 2>&1 &

# Argument separate:
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-84-drug-uw.yaml > ./ner_results/train_ner_drug_clinical.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-84-alcohol-uw.yaml > ./ner_results/train_ner_alcohol_clinical.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer.py sdoh-84-tobacco-uw.yaml > ./ner_results/train_ner_tobacco_clinical.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-84-livingstatus-uw.yaml > ./ner_results/train_ner_liv_clinicalp.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-84-employment-uw.yaml > ./ner_results/train_ner_emp_clinical.out 2>&1 &



# argument ner model training

# option 1 together:
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer.py sdoh-26-arg-together-uw.yaml > ./ner_results/train_arg_together_uw.out 2>&1 &

# option 2 separate:
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-drug-uw.yaml > ./ner_results/train_ner_drug_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-alcohol-uw.yaml > ./ner_results/train_ner_alcohol.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-tobacco-uw.yaml > ./ner_results/train_ner_tobacco.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer.py sdoh-26-livingstatus-uw.yaml > ./ner_results/train_ner_liv_overlap.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-employment-uw.yaml > ./ner_results/train_ner_emp_overlap.out 2>&1 &



# Relation Extraction: Classification - Filter Relation

# Data Preprocessing
# train
python relation_ps.py sdoh-26-event.yaml # generate positive example  - match 		save: relation_train_match.csv
python relation_ne.py sdoh-26-event.yaml # generate negative example - not match 	save: relation_train_notmatch.csv
# dev
python relation_ps.py sdoh-26-event-dev.yaml	    #save: relation_dev_match.csv
python relation_ne.py sdoh-26-event-dev.yaml    #save: relation_dev_notmatch.csv
# combine positive and negative to #trainordev
cat relation_train_match.csv relation_train_notmatch.csv > relation_train.csv	# train: relation_train.csv
cat relation_dev_match.csv relation_dev_notmatch.csv > relation_dev.csv			# dev: relation_dev.csv

# Train the Match Classification Model
nohup python relation_pcl.py > ./relation_results/train_relation_uw.out 2>&1 & 
nohup python relation_pcl.py > ./relation_results/train_relation_123.out 2>&1 & 

# Relation Extraction: Classification - Filter Relation

# Data Preprocessing for Classification
# train
python relation_ps.py sdoh-26-event.yaml # generate positive example  - match 		save: relation_train_match.csv
python relation_ne.py sdoh-26-event.yaml # generate negative example - not match 	save: relation_train_notmatch.csv

# dev
python relation_ps.py sdoh-26-event-dev.yaml	    #save: relation_dev_match.csv
python relation_ne.py sdoh-26-event-dev.yaml    #save: relation_dev_notmatch.csv

# combine positive and negative sample to train/dev
cat relation_train_match.csv relation_train_notmatch.csv > relation_train.csv	# train: relation_train.csv
cat relation_dev_match.csv relation_dev_notmatch.csv > relation_dev.csv			# dev: relation_dev.csv



# Argument Subtype Classification: Know distribution(template_rl, train, dev)

python relation_subtype.py train		# train 10933, 3512 981 959 959 
python relation_subtype.py dev			# dev 1177, 416 90 117 117

# move train and dev to template_rl
bash move_subtype_data.sh


# train the model
# train_data, dev_data, test_data, where_to_save_best_model, device_cuda
nohup python argument_subtype_pcl.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_dev_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:1' > ./relation_results/train_subtype_med_123.out 2>&1 & 
# subtype_train_med.csv subtype_dev_med.csv # save model

nohup python argument_subtype_pcl.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:2' > ./relation_results/train_subtype_emp_123.out 2>&1 & 
# subtype_train_emp.csv subtype_dev_emp.csv #

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:1' > ./relation_results/train_subtype_liv_type_123.out 2>&1 & 
# subtype_train_liv_type.csv subtype_dev_liv_type.csv #

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' > ./relation_results/train_subtype_liv_status_123.out 2>&1 &
# subtype_train_liv_status.csv subtype_dev_liv_status.csv #
