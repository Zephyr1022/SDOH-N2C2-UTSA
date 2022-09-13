# NER-Trigger -> NER-Argument -> Relation Classification -> Subtype Classification
# Pred               Fixed             Pred                        Pred 


rm ./ANNTABLE/system5/ann/*
rm ./ANNTABLE/system5/table/*
rm ./ANNTABLE/system5/argu_drug/*
rm ./ANNTABLE/system5/argu_alcohol/*
rm ./ANNTABLE/system5/argu_tobacco/*
rm ./ANNTABLE/system5/argu_emp/*
rm ./ANNTABLE/system5/argu_liv/*
rm ./ANNTABLE/system5/piece_relation/*


##############################################################
# Trigger NER - 2-notag Pred
##############################################################        
# yaml_model_name, input_test, output_tes; final-model.pt or best-model.pt
CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-26-trigger-piece.yaml' 'test2_triggers_ner.txt' 'test2_triggers_pred.txt' > ./ner_results/trigger_ner_testsys1.out 2>&1 &

# clean ann table
rm ./ANNTABLE/system5/ann/*.ann

# generate predicted trigger ann (T#) and prediction
# ann_save_dir, conll_order, above_pred, output_pred
python test2ann_events.py './ANNTABLE/system5/ann/' './conll_num/test2_triggers_num.conll' 'test2_triggers_pred.txt' 'test2_triggers_relation.txt'

##############################################################
# Argument NER - 2-seperate Fixed 
##############################################################        
# generate all groundtruth ann for argument
python argument_extract_piece.py Drug argu_drug
python argument_extract_piece.py Alcohol argu_alcohol
python argument_extract_piece.py Tobacco argu_tobacco
python argument_extract_piece.py LivingStatus argu_liv
python argument_extract_piece.py Employment argu_emp


mv test2_triggers_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Drug_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Alcohol_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Tobacco_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_LivingStatus_relation.txt ./ANNTABLE/system5/piece_relation
mv test2_Employment_relation.txt ./ANNTABLE/system5/piece_relation

##############################################################
# Generate all-poss-relations-trigger-argument & subtype test data 
##############################################################

# Test Data Prepare
python match_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt' 'relation_test22_piece.csv' './relation_pred/test22-piece-trigger-argument-all-poss-relation.txt' 

# Match Prediction
python relation_pcl_pred.py 'relation_test22_piece.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' 'relation_pred/testsys1-match-pred-V123.csv' 'relation_pred/testsys1-match-prob-V123.csv'

# Subtype Test Data Prepare
python subtype_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt'

# OUTPUT for classification prediction 
 # 64     input_file0 = './ANNTABLE/system5/piece_relation/test2_triggers_relation.txt' #triggers
 # 65     input_file1 = './ANNTABLE/system5/piece_relation/test2_Drug_relation.txt' #Drug 
 # 66     input_file2 = './ANNTABLE/system5/piece_relation/test2_Alcohol_relation.txt' #Alcohol
 # 67     input_file3 = './ANNTABLE/system5/piece_relation/test2_Tobacco_relation.txt' #Tobacco
 # 68     input_file4 = './ANNTABLE/system5/piece_relation/test2_LivingStatus_relation.txt' #LivingStatus
 # 69     input_file5 = './ANNTABLE/system5/piece_relation/test2_Employment_relation.txt' #Employment
# 208     df_med.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_med.csv', header = False) # relative position
# 209     df_emp.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_emp.csv', header = False) # relative position
# 210     df_liv_Status.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_liv_status.csv', header = False) # relative position
# 211     df_liv_Type.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_liv_type.csv', header = False) # relative position

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
vim scoring_testsys2.csv


# F1
OVERALL,OVERALL,OVERALL,3471,3507,3212,0.9158825206729398,0.9253817343704984,0.9206076239610204

# relation_pred
test11-base-match-pred-V123.csv              test22-base-pred-subtype-emp-123.csv                 test22-uw-base-match-prob-V123.csv
test11-base-match-prob-V123.csv              test22-base-pred-subtype-liv-status-123.csv          testsys1-base-pred-subtype-emp-123.csv
test11-base-pred-subtype-emp-123.csv         test22-base-pred-subtype-liv-type-123.csv            testsys1-base-pred-subtype-liv-status-123.csv
test11-base-pred-subtype-liv-status-123.csv  test22-base-pred-subtype-med-123.csv                 testsys1-base-pred-subtype-liv-type-123.csv
test11-base-pred-subtype-liv-type-123.csv    test22-mimic-base-match-pred-V123.csv                testsys1-base-pred-subtype-med-123.csv
test11-base-pred-subtype-med-123.csv         test22-mimic-base-match-prob-V123.csv                testsys1-match-pred-V123.csv
test22-base-match-pred-V123.csv              test22-piece-trigger-argument-all-poss-relation.txt  testsys1-match-prob-V123.csv
test22-base-match-prob-V123.csv              test22-uw-base-match-pred-V123.csv