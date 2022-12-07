# Step 3, generate predicted trigger ann (T#) and prediction
python test2ann_events.py './experiments/system5/ann/' './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_triggers_pred.txt' 'test1_triggers_relation.txt'

# Step 4, Generate arguments-relation (leave-one-behind issue for conll)
python test2ann_arguments.py './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Drug_relation.txt'
python test2ann_arguments.py './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Alcohol_relation.txt'
python test2ann_arguments.py './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Tobacco_relation.txt'

python test2ann_arguments.py './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_Employment_arguments_pred.txt' 'test1_Employment_relation.txt'
python test2ann_arguments.py './taskb/ner_pred_triaffine/triaffine_joint.conll' './taskb/ner_pred_triaffine/triaffine_LivingStatus_arguments_pred.txt' 'test1_LivingStatus_relation.txt'

# pred_trigger.write(dataset[snt][chr_str-1][-1]+' '+ data[1] + ' '+ ' '.join(dataset[snt][chr_str-1][2:4]) +' '+ ' '.join(data[3:])+' '+'\n')
# KeyError: 1163



# Step 5-trigger, argument, Archive/归档
mv test1_triggers_relation.txt ./experiments/system5/piece_relation
mv test1_Drug_relation.txt ./experiments/system5/piece_relation
mv test1_Alcohol_relation.txt ./experiments/system5/piece_relation
mv test1_Tobacco_relation.txt ./experiments/system5/piece_relation
mv test1_LivingStatus_relation.txt ./experiments/system5/piece_relation
mv test1_Employment_relation.txt ./experiments/system5/piece_relation









# Classification - RoBerta

# Step 6, Relation Classification Test Data
python match_relation_unif.py './Annotations/test_sdoh/taskb/*.txt' './relation_pred/relation_classification_temp.csv' './relation_pred/sys11-taskb-t53b-trigger-argument-all-poss-relation.txt' 'test1' 'test1'

# Subtype Test Data Prepare, gdtruth_txt, trigger_num, argument_sys, subtype_sys, subtype_name
python subtype_relation_unif.py './Annotations/test_sdoh/taskb/*.txt' 'test1' 'test1' 'temp'

# Step 7, Relation Classification Prediction rely on the NER above
# Sybtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './experiments/system5/piece_subtype/subtype_temp_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' './relation_pred/temp-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './experiments/system5/piece_subtype/subtype_temp_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' './relation_pred/temp-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './experiments/system5/piece_subtype/subtype_temp_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' './relation_pred/temp-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './experiments/system5/piece_subtype/subtype_temp_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' './relation_pred/temp-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type-sys11-taska.out 2>&1 &

# Match Prediction
python relation_pcl_pred.py './relation_pred/relation_classification_temp.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:1' './relation_pred/temp-match-pred-V123.csv' 'relation_pred/temp-match-prob-V123.csv'


# Step 8, Ensemble SDOH Events 373
conda activate sdohV1

# ls -1q *.txt | wc -l
# ls -1q *.ann | wc -l

rm ~/sdoh/experiments/system5/table/*.ann
rm ~/sdoh/experiments/system5/table/*.txt
cp ~/sdoh/Annotations/test_sdoh/taskb/*.txt ~/sdoh/experiments/system5/table/
cp ~/sdoh/experiments/system5/ann/*.ann ~/sdoh/experiments/system5/table/ 


# Match Prediction: ground_truth_relation.txt, testsys1-relation.txt
python relation_cls_gt.py
python cls_pred_gt.py


# option 4: Binary Filter + threshold + argmax   
python relation_match_argmax.py './relation_pred/sys11-taskb-t53b-trigger-argument-all-poss-relation.txt' './relation_pred/temp-match-prob-V123.csv' 'temp-argmax-threshold-relation.txt' 0.1

# ensemble table: including subtype prediction, mimic_table22.py missing one type for substance use
python mimic_table22.py 'temp-argmax-threshold-relation.txt' 'system5'

# get scotes
python get_temp_results.py "test_sdoh/taskb" "system5" "taskb-triaffine-roberta-joint" 

vim scoring_taskb-triaffine-roberta-joint.csv
