# Record Best Model for flair_search26

# sdoh_flair_search26_trigger_tag_uw
# sdoh_flair_search26_trigger_uw 
# saving best model

##############################################################
# Active virtual environment
##############################################################

conda activate scispacyV5

##############################################################
# Trigger w/o tag
##############################################################

# sdoh_flair_search26_event_uw
# EPOCH 14 done: loss 0.0391 - lr 0.0500000
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-trigger-mimic_train_dev.yaml train_trigger_ner.txt dev_trigger_ner.txt > ./ner_results/trigger_ner_mimic_train_dev.out 2>&1 &

##############################################################
# Trigger w/ tag
##############################################################

# sdoh_flair_search26_event_tag_uw
# EPOCH 21 done: loss 0.0058 - lr 0.0250000
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer2.py sdoh-26-trigger-tag-uw_train_dev.yaml train_triggers_tag_ner.txt dev_triggers_tag_ner.txt > ./ner_results/trigger_tag_ner_uw_train_dev.out 2>&1 &

##############################################################
# Argument Together***
##############################################################

CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-arg-together-uw.yaml train_arg_together_uw_ner.txt dev_arg_together_uw_ner.txt  > ./ner_results/arg_together_ner_mimic_train_dev.out 2>&1 &

##############################################################
# Argument Seperate
#        Drug, Alcohol, Tobacco, Livingstatue, and Employment
##############################################################

# Drug
# EPOCH 8 done: loss 0.0110 - lr 0.1000000 DEV : loss 0.008454359136521816 - f1-score (micro avg)  0.8093

CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer2.py sdoh-26-drug-mimic_train_dev.yaml train_argu_drug_uw_ner.txt dev_argu_drug_uw_ner.txt > ./ner_results/drug_ner_mimic_train_dev.out 2>&1 &

# Alcohol
# EPOCH 9 done: loss 0.0094 - lr 0.1000000; DEV : loss 0.008061129599809647 - f1-score (micro avg)  0.7855

CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer2.py sdoh-26-alcohol-mimic_train_dev.yaml train_argu_alcohol_uw_ner.txt dev_argu_alcohol_uw_ner.txt > ./ner_results/alcohol_ner_mimic_train_dev.out 2>&1 &

# Tobacco
# EPOCH 16 done: loss 0.0052 - lr 0.0500000; DEV : loss 0.01181077491492033 - f1-score (micro avg)  0.8187

CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer2.py sdoh-26-tobacco-mimic_train_dev.yaml train_argu_tobacco_uw_ner.txt dev_argu_tobacco_uw_ner.txt  > ./ner_results/tobacco_ner_mimic_train_dev.out 2>&1 &

# Livingstatue
# EPOCH 14 done: loss 0.0053 - lr 0.0250000; DEV : loss 0.028041338548064232 - f1-score (micro avg)  0.8296

CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-livingstatus-mimic_train_dev.yaml train_argu_liv_uw_ner.txt dev_argu_liv_uw_ner.txt > ./ner_results/liv_ner_mimic_train_dev.out 2>&1 &

# Employment
# EPOCH 12 done: loss 0.0050 - lr 0.0500000; DEV : loss 0.022013071924448013 - f1-score (micro avg)  0.7079

CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer2.py sdoh-26-employment-mimic_train_dev.yaml train_argu_emp_uw_ner.txt dev_argu_emp_uw_ner.txt > ./ner_results/emp_ner_mimic_train_dev.out 2>&1 &


##############################################################
#            Train the Match Classification Model
##############################################################

# OUTPUT_MODEL_PATH  = './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' (./model_save)

# Train the Match Classification Model
# train_data, dev_data, where_to_save_best_model, device_cuda

nohup python relation_pcl.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' > ./relation_results/match_relation_testsys1.out 2>&1 & 


##############################################################
#            Train the Subtype Classification Model
##############################################################

# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt'  (./model_save) 这个UW没有训练
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt'


# Train the argument_subtype model (test = dev)
# train_data, dev_data, test_data, where_to_save_best_model, device_cuda

nohup python argument_subtype_pcl.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_dev_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' > ./relation_results/train_subtype_med_123.out 2>&1 & 

nohup python argument_subtype_pcl.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' > ./relation_results/train_subtype_emp_123.out 2>&1 & 

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' > ./relation_results/train_subtype_liv_type_123.out 2>&1 & 

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' > ./relation_results/train_subtype_liv_status_123.out 2>&1 &


# subtype_train_med.csv subtype_dev_med.csv # save model
# subtype_train_emp.csv subtype_dev_emp.csv
# subtype_train_liv_type.csv subtype_dev_liv_type.csv
# subtype_train_liv_status.csv subtype_dev_liv_status.csv