# For Task C - Best MODEL T53B

# Step 1, Trigger NER - <Trigger><Drug>
CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_triggers_ner.txt' './test_pred/test1_triggers_pred.txt' > ./ner_results/event_trigger_t5_sys1_taskc.out 2>&1 &

# Step 2, Argument NER - Joint <Argument><Drug><Type>
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_Drug_ner.txt' './test_pred/test1_Drug_arguments_pred.txt' > ./ner_results/event_argument_ner_drug_t5_all_sys1_taskc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_Alcohol_ner.txt' './test_pred/test1_Alcohol_arguments_pred.txt' > ./ner_results/event_argument_ner_alcohol_t5_all_sys1_taskc.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_Tobacco_ner.txt' './test_pred/test1_Tobacco_arguments_pred.txt' > ./ner_results/event_argument_ner_tobacco_t5_all_sys1_taskc.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_Employment_ner.txt' './test_pred/test1_Employment_arguments_pred.txt' > ./ner_results/event_argument_ner_employment_t5_all_sys1_taskc.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-biobert-joint-event-mimic-uw.yaml' 'test1_LivingStatus_ner.txt' './test_pred/test1_LivingStatus_arguments_pred.txt' > ./ner_results/event_argument_ner_livingstatus_t5_all_sys1_taskc.out 2>&1 &

# Step 3, ann & relation 换到 UTSA server 再 run
python test2ann_events.py './experiments/system5/ann/' './conll_num/test1_triggers_num.conll' './test_pred/test1_triggers_pred.txt' 'test1_triggers_relation.txt'

# Step 4, Generate arguments-relation (leave-one-behind issue for conll)
python test2ann_arguments.py './conll_num/test1_Drug_num.conll' './test_pred/test1_Drug_arguments_pred.txt' 'test1_Drug_relation.txt'
python test2ann_arguments.py './conll_num/test1_Alcohol_num.conll' './test_pred/test1_Alcohol_arguments_pred.txt' 'test1_Alcohol_relation.txt'
python test2ann_arguments.py './conll_num/test1_Tobacco_num.conll' './test_pred/test1_Tobacco_arguments_pred.txt' 'test1_Tobacco_relation.txt'
python test2ann_arguments.py './conll_num/test1_Employment_num.conll' './test_pred/test1_Employment_arguments_pred.txt' 'test1_Employment_relation.txt'
python test2ann_arguments.py './conll_num/test1_LivingStatus_num.conll' './test_pred/test1_LivingStatus_arguments_pred.txt' 'test1_LivingStatus_relation.txt'

# Step 5, Archive
mv test1_triggers_relation.txt ./experiments/system5/piece_relation
mv test1_Drug_relation.txt ./experiments/system5/piece_relation
mv test1_Alcohol_relation.txt ./experiments/system5/piece_relation
mv test1_Tobacco_relation.txt ./experiments/system5/piece_relation
mv test1_LivingStatus_relation.txt ./experiments/system5/piece_relation
mv test1_Employment_relation.txt ./experiments/system5/piece_relation

# Step 6, Relation Classification Test Data Prepare
python match_relation_unif.py './Annotations/test_sdoh/uw/*.txt' './relation_pred/relation_classification_temp.csv' './relation_pred/trigger-argument-all-poss-relation.txt' 'test1' 'test1'

# Subtype Test Data Prepare
python subtype_relation_unif.py './Annotations/test_sdoh/uw/*.txt' 'test1' 'test1' 'temp'


# Step 7, Relation Classification Prediction rely on the NER above
# Sybtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './experiments/system5/piece_subtype/subtype_temp_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' './relation_pred/temp-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './experiments/system5/piece_subtype/subtype_temp_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' './relation_pred/temp-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './experiments/system5/piece_subtype/subtype_temp_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' './relation_pred/temp-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './experiments/system5/piece_subtype/subtype_temp_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' './relation_pred/temp-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type-sys11-taska.out 2>&1 &

# Match Prediction
python relation_pcl_pred.py './relation_pred/relation_classification_temp.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:1' './relation_pred/temp-match-pred-V123.csv' 'relation_pred/temp-match-prob-V123.csv'

# Step 8, Ensemble SDOH Events
conda activate sdohV1

rm ~/sdoh/experiments/system5/table/*.ann
rm ~/sdoh/experiments/system5/table/*.txt
cp ~/sdoh/Annotations/test_sdoh/uw/*.txt ~/sdoh/experiments/system5/table/
cp ~/sdoh/experiments/system5/ann/*.ann ~/sdoh/experiments/system5/table/ 

# Binary Filter + threshold + argmax   
python relation_match_argmax.py './relation_pred/trigger-argument-all-poss-relation.txt' './relation_pred/temp-match-prob-V123.csv' 'temp-argmax-threshold-relation.txt' 0.1

# ensemble table: including subtype prediction
python all_table_temp.py 'temp-argmax-threshold-relation.txt' 'system5'

# get scotes
python get_temp_results.py "test_sdoh/uw" "system5" "taskc-t53b-biobert-joint" 
