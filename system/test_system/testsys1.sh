# NER-Trigger -> NER-Argument -> Relation Classification -> Subtype Classification
# Fixed               Pred             Pred                        Pred 

# reset system5
rm ./ANNTABLE/system5/ann/*
rm ./ANNTABLE/system5/table/*
rm ./ANNTABLE/system5/argu_drug/*
rm ./ANNTABLE/system5/argu_alcohol/*
rm ./ANNTABLE/system5/argu_tobacco/*
rm ./ANNTABLE/system5/argu_emp/*
rm ./ANNTABLE/system5/argu_liv/*
rm ./ANNTABLE/system5/piece_relation/*

conda activate scispacyV5


# generate all groundtruth ann for trigger
python event_trigger_piece.py system5


##############################################################
# Argument NER - 2-seperate
############################################################## 
# input_text: best_model, test_data, output_test, final-model.pt or best-model.pt

CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-drug-piece.yaml' 'test2_Drug_ner.txt' 'test2_Drug_arguments_pred.txt' > ./ner_results/argument_ner_drug_testsys1.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-26-alcohol-piece.yaml' 'test2_Alcohol_ner.txt' 'test2_Alcohol_arguments_pred.txt' > ./ner_results/argument_ner_alcohol_testsys1.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-26-tobacco-piece.yaml' 'test2_Tobacco_ner.txt' 'test2_Tobacco_arguments_pred.txt' > ./ner_results/argument_ner_tobacco_testsys1.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-employment-piece.yaml' 'test2_Employment_ner.txt' 'test2_Employment_arguments_pred.txt' > ./ner_results/argument_ner_employment_testsys1.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-livingstatus-piece.yaml' 'test2_LivingStatus_ner.txt' 'test2_LivingStatus_arguments_pred.txt' > ./ner_results/argument_ner_livingstatus_testsys1.out 2>&1 &

# Generate arguments-relation 
python test2ann_arguments.py './conll_num/test2_Drug_num.conll' 'test2_Drug_arguments_pred.txt' 'test2_Drug_relation.txt'
python test2ann_arguments.py './conll_num/test2_Alcohol_num.conll' 'test2_Alcohol_arguments_pred.txt' 'test2_Alcohol_relation.txt'
python test2ann_arguments.py './conll_num/test2_Tobacco_num.conll' 'test2_Tobacco_arguments_pred.txt' 'test2_Tobacco_relation.txt'
python test2ann_arguments.py './conll_num/test2_Employment_num.conll' 'test2_Employment_arguments_pred.txt' 'test2_Employment_relation.txt'
python test2ann_arguments.py './conll_num/test2_LivingStatus_num.conll' 'test2_LivingStatus_arguments_pred.txt' 'test2_LivingStatus_relation.txt'

mv test2_triggers_pred.txt test_pred
mv test2_Drug_arguments_pred.txt test_pred
mv test2_Alcohol_arguments_pred.txt test_pred
mv test2_Tobacco_arguments_pred.txt test_pred
mv test2_Employment_arguments_pred.txt test_pred
mv test2_LivingStatus_arguments_pred.txt test_pred

##############################################################
# Generate all-poss-relations-trigger-argument & subtype test data 
##############################################################

mv test2_triggers_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Drug_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Alcohol_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Tobacco_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_LivingStatus_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Employment_relation.txt ./ANNTABLE/system5/piece_relation

# Test Data Prepare
python match_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt' 'relation_test22_piece.csv' './relation_pred/test22-piece-trigger-argument-all-poss-relation.txt' 

# Match Prediction
python relation_pcl_pred.py 'relation_test22_piece.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' 'relation_pred/testsys1-match-pred-V123.csv' 'relation_pred/testsys1-match-prob-V123.csv'

# Subtype Test Data Prepare
python subtype_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt'

# Sybtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
# train_data, dev_data, test_data, best_model, device_cuda, result_save
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './ANNTABLE/system5/piece_subtype/subtype_test22_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' 'relation_pred/testsys1-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_testsys1.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './ANNTABLE/system5/piece_subtype/subtype_test22_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' 'relation_pred/testsys1-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_testsys1.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './ANNTABLE/system5/piece_subtype/subtype_test22_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' 'relation_pred/testsys1-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_testsys1.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './ANNTABLE/system5/piece_subtype/subtype_test22_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' 'relation_pred/testsys1-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_testsys1.out 2>&1 &

##############################################################
# Ensemble Events
##############################################################

# activate virtual environment 
conda activate sdohV1

# reset table
rm ~/sdoh/ANNTABLE/system5/table/*.ann
rm ~/sdoh/ANNTABLE/system5/table/*.txt
cp ~/sdoh/Annotations/test_sdoh/mimic/*.txt ~/sdoh/ANNTABLE/system5/table/
cp ~/sdoh/ANNTABLE/system5/ann/*.ann ~/sdoh/ANNTABLE/system5/table/ 

# Filter + threshold + argmax   
python relation_match_argmax.py './relation_pred/test22-piece-trigger-argument-all-poss-relation.txt'  './relation_pred/testsys1-match-prob-V123.csv' "testsys1-argmax-threshold-relation.txt" 0.01

# ensemble table
python piece_table22.py 'testsys1-argmax-threshold-relation.txt' 'system5'

# get scotes
python get_results_testsys1.py
vim scoring_testsys1.csv


# F1
OVERALL,OVERALL,OVERALL,3471,3074,2874,0.9349381912817176,0.8280034572169404,0.8782276546982429


