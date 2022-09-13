# Record Best Model for flair_search26

##############################################################
# Active virtual environment
##############################################################

conda activate scispacyV5

##############################################################
# Trigger w/o tag (./tagger)
##############################################################

CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-trigger-piece.yaml train_trigger_ner.txt dev_trigger_ner.txt > ./ner_results/trigger_ner_piece.out 2>&1 &

##############################################################
# Trigger w/ tag
##############################################################


CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-trigger-tag-piece.yaml train_triggers_tag_ner.txt dev_triggers_tag_ner.txt > ./ner_results/trigger_tag_ner_piece.out 2>&1 &

##############################################################
# Argument Together***
##############################################################

CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer2.py sdoh-26-arg-together-piece.yaml train_arg_together_uw_ner.txt dev_arg_together_uw_ner.txt > ./ner_results/arg_together_ner_piece.out 2>&1 &

##############################################################
# Argument Seperate
#        Drug, Alcohol, Tobacco, Livingstatue, and Employment
##############################################################

# Drug
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer2.py sdoh-26-drug-piece.yaml train_argu_drug_uw_ner.txt dev_argu_drug_uw_ner.txt > ./ner_results/drug_ner_piece.out 2>&1 &

# Alcohol
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-alcohol-piece.yaml train_argu_alcohol_uw_ner.txt dev_argu_alcohol_uw_ner.txt > ./ner_results/alcohol_ner_piece.out 2>&1 &

# Tobacco
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer2.py sdoh-26-tobacco-piece.yaml train_argu_tobacco_uw_ner.txt dev_argu_tobacco_uw_ner.txt > ./ner_results/tobacco_ner_piece.out 2>&1 &

# Livingstatue
CUDA_VISIBLE_DEVICES=2 nohup python sdoh_trainer2.py sdoh-26-livingstatus-piece.yaml train_argu_liv_uw_ner.txt dev_argu_liv_uw_ner.txt > ./ner_results/liv_ner_piece.out 2>&1 &

# Employment
CUDA_VISIBLE_DEVICES=3 nohup python sdoh_trainer2.py sdoh-26-employment-piece.yaml train_argu_emp_uw_ner.txt dev_argu_emp_uw_ner.txt > ./ner_results/emp_ner_piece.out 2>&1 &


##############################################################
#     Train the Match Classification Model (./model_save)
##############################################################
# Train the Match Classification Model
# train_data, dev_data, where_to_save_best_model, device_cuda

nohup python relation_pcl.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' > ./relation_results/match_relation_testsys1.out 2>&1 & 

# OUTPUT_MODEL_PATH  = './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' # match
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt'  # 这个UW没有训练
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 
# OUTPUT_MODEL_PATH = './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt'

##############################################################
#      Train the Subtype Classification Model (./model_save)
##############################################################
# Train the argument_subtype model
# train_data, dev_data, test_data, where_to_save_best_model, device_cuda:

nohup python argument_subtype_pcl.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' > ./relation_results/tiriain_subtype_med_testsys1.out 2>&1 & 

nohup python argument_subtype_pcl.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' > ./relation_results/train_subtype_emp_testsys1.out 2>&1 & 

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_testsys1.out 2>&1 &

nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_testsys1.out 2>&1 & 

# Need in the prediction stage
# './model_save/distilbert-model-match-testsys1.pt'
# './model_save/distilbert-subtype-med-testsys1.pt'
# './model_save/distilbert-subtype-emp-testsys1.pt'
# './model_save/distilbert-subtype-liv-status-testsys1.pt'
# './model_save/distilbert-subtype-liv-type-testsys1.pt'
