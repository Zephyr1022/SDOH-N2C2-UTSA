##############################################################
#         
#           Extracting Social Determinants of Health 
#
#                    Subtask A: Extraction 
#                    Subtask B: Generalizability      
#                    Subtask C: Learning Transfer 
##############################################################

# Active virtual environment
conda activate scispacyV5
conda activate sdoh_n2c2
conda activate sdohV1
conda activate sdoh_flair

# check results
%!grep "f1-score (micro avg)"
%!grep "Trigger"
%!grep "Alcohol"
%!grep "Drug"
%!grep "Tobacco"
%!grep "Employment"
%!grep "LivingStatus"

# NT - count of true (gold) labels
# NP - count of predicted labels
# TP - counted true positives


##############################################################
#       STEP I: Train the NER Model (./tagger)
##############################################################

# Option 1. NER model for joint trigger and argument - Flair 
# task a 
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-joint-event-mimic.yaml mimic_train_ner.txt mimic_dev_ner.txt > ./ner_results/joint_event_ner_flair_mimic.out 2>&1 &
# task c 
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer2.py sdoh-26-joint-event-mimic-uw.yaml train_ner.txt dev_ner.txt > ./ner_results/joint_event_ner_flair_mimic_uw.out 2>&1 &


# Option 2. NER model for joint trigger and argument - T5-3B (Best Model)
# task a
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer3.py sdoh-84-joint-event-mimic.yaml mimic_train_ner.txt mimic_dev_ner.txt > ./ner_results/joint_event_ner_t5_mimic.out 2>&1 &
# task c
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer3.py sdoh-84-joint-event-mimic-uw.yaml mimic-uw_train_ner.txt mimic-uw_dev_ner.txt > ./ner_results/joint_event_ner_t5_mimic_uw.out 2>&1 &


# Option 3. NER model for joint trigger and argument - BioBert
# task a
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer_biobert.py sdoh-biobert-joint-event-mimic.yaml mimic_train_ner.txt mimic_dev_ner.txt > ./ner_results/mimic_biobert_joint_event_ner.out 2>&1 &
# task c 
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer_biobert.py sdoh-biobert-joint-event-mimic-uw.yaml mimic-uw_train_ner.txt mimic-uw_dev_ner.txt > ./ner_results/mimic_uw_biobert_joint_event_ner.out 2>&1 &


# Option 4. New Marker Schema only applied on BEST T53B model 
# task a
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer3.py sdoh-t53b-joint-event-mimic-new.yaml new_mimic_train_ner.txt new_mimic_dev_ner.txt > ./ner_results/joint_event_ner_t5_new_mimic.out 2>&1 &
# task c
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer3.py sdoh-t53b-joint-event-mimic-uw-new.yaml new_mimic_uw_train_ner.txt new_mimic_uw_dev_ner.txt > ./ner_results/joint_event_ner_t5_new_mimic_uw.out 2>&1 &



#################################################################
#    STEP II: Train the Match Classification Model (./model_save)
#################################################################

# Task A - RoBerta
nohup python relation_pcl.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/distilbert-model-match-mimic.pt' 'cuda:0' > ./relation_results/match_relation_roberta_mimic.out 2>&1 & 

# Task C - RoBerta
nohup python relation_pcl.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/distilbert-model-match-all.pt' 'cuda:1' > ./relation_results/match_relation_roberta_all.out 2>&1 & 


# Task A - ClinicalBert
nohup python relation_clinicalbert.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/clinicalbert-model-match-mimic.pt' 'cuda:0' > ./relation_results/match_relation_clinical_mimic.out 2>&1 & 

# Task C - ClinicalBert
nohup python relation_clinicalbert.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/clinicalbert-model-match-all.pt' 'cuda:1' > ./relation_results/match_relation_clinical_all.out 2>&1 & 


# Task A - BioBert
CUDA_LAUNCH_BLOCKING=0 nohup python relation_biobert.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/biobert-model-match-mimic.pt' 'cuda:0' > ./relation_results/match_relation_bio_mimic.out 2>&1 & 

# Task C - BioBert
CUDA_VISIBLE_DEVICES=0 nohup python relation_biobert.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/biobert-model-match-all.pt' 'cuda:0' > ./relation_results/match_relation_bio_all.out 2>&1 & 


# Task A - T5-3B
CUDA_LAUNCH_BLOCKING=0 nohup python relation_t53b2.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/t53b-model-match-mimic.pt' 'cuda:0' > ./relation_results/match_relation_t53b_mimic.out 2>&1 & 

# Task C - T5-3B
CUDA_VISIBLE_DEVICES=0 nohup python relation_t53b2.py './template_rl/relation_train.csv' './template_rl/relation_dev.csv' './model_save/t53b-model-match-all.pt' 'cuda:0' > ./relation_results/match_relation_t53b_all.out 2>&1 & 


#################################################################
#  STEP III: Train the Subtype Classification Model (./model_save)
#################################################################

# Task A - RoBerta
nohup python argument_subtype_pcl.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/distilbert-subtype-med-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_med_mimic.out 2>&1 & 
nohup python argument_subtype_pcl.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/distilbert-subtype-emp-mimic.pt' 'cuda:1' > ./relation_results/train_subtype_emp_mimic.out 2>&1 & 
nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/distilbert-subtype-liv-status-mimic.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_mimic.out 2>&1 &
nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/distilbert-subtype-liv-type-mimic.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_mimic.out 2>&1 & 


# Task C - RoBerta
nohup python argument_subtype_pcl.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/distilbert-subtype-med-all.pt' 'cuda:0' > ./relation_results/tiriain_subtype_med_all.out 2>&1 & 
nohup python argument_subtype_pcl.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/distilbert-subtype-emp-all.pt' 'cuda:1' > ./relation_results/train_subtype_emp_all.out 2>&1 & 
nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/distilbert-subtype-liv-status-all.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_all.out 2>&1 &
nohup python argument_subtype_pcl.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/distilbert-subtype-liv-type-all.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_all.out 2>&1 & 


# Task A - ClinicalBert
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/clinicalbert-subtype-med-mimic.pt' 'cuda:0' > ./relation_results/tiriain_subtype_med_clinicalbert_mimic.out 2>&1 & 
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/clinicalbert-subtype-emp-mimic.pt' 'cuda:1' > ./relation_results/train_subtype_emp_clinicalbert_mimic.out 2>&1 & 
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/clinicalbert-subtype-liv-status-mimic.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_clinicalbert_mimic.out 2>&1 &
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/clinicalbert-subtype-liv-type-mimic.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_clinicalbert_mimic.out 2>&1 & 

# Task C - ClinicalBert
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/clinicalbert-subtype-med-all.pt' 'cuda:0' > ./relation_results/tiriain_subtype_med_clinicalbert_all.out 2>&1 & 
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/clinicalbert-subtype-emp-all.pt' 'cuda:1' > ./relation_results/train_subtype_emp_clinicalbert_all.out 2>&1 & 
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/clinicalbert-subtype-liv-status-all.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_clinicalbert_all.out 2>&1 &
nohup python argument_subtype_clinicalbert.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/clinicalbert-subtype-liv-type-all.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_clinicalbert_all.out 2>&1 & 


# Task A - BioBert
nohup python argument_subtype_biobert.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './model_save/biobert-subtype-med-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_med_biobert_mimic.out 2>&1 & 
nohup python argument_subtype_biobert.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './model_save/biobert-subtype-emp-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_emp_biobert_mimic.out 2>&1 & 
nohup python argument_subtype_biobert.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './model_save/biobert-subtype-liv-status-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_liv_status_biobert_mimic.out 2>&1 &
nohup python argument_subtype_biobert.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './model_save/biobert-subtype-liv-type-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_liv_type_biobert_mimic.out 2>&1 & 

# Task C - BioBert in Server UTSA 
nohup python argument_subtype_biobert.py './template_rl/taskc-relation-temp/subtype_train_med.csv' './template_rl/taskc-relation-temp/subtype_dev_med.csv' './model_save/biobert-subtype-med-mimic.pt' 'cuda:0' > ./relation_results/train_subtype_med_biobert_taskc.out 2>&1 & 
nohup python argument_subtype_biobert.py './template_rl/taskc-relation-temp/subtype_train_emp.csv' './template_rl/taskc-relation-temp/subtype_dev_emp.csv' './model_save/biobert-subtype-emp-mimic.pt' 'cuda:1' > ./relation_results/train_subtype_emp_biobert_taskc.out 2>&1 & 
nohup python argument_subtype_biobert.py './template_rl/taskc-relation-temp/subtype_train_liv_status.csv' './template_rl/taskc-relation-temp/subtype_dev_liv_status.csv' './model_save/biobert-subtype-liv-status-mimic.pt' 'cuda:2' > ./relation_results/train_subtype_liv_status_biobert_taskc.out 2>&1 &
nohup python argument_subtype_biobert.py './template_rl/taskc-relation-temp/subtype_train_liv_type.csv' './template_rl/taskc-relation-temp/subtype_dev_liv_type.csv' './model_save/biobert-subtype-liv-type-mimic.pt' 'cuda:3' > ./relation_results/train_subtype_liv_type_biobert_taskc.out 2>&1 & 










