# NER-Trigger -> NER-Argument -> Subtype Classification  -> Relation Classification 
# Pred               Pred             Fixed                           Pred                       


rm ./ANNTABLE/system5/ann/*
rm ./ANNTABLE/system5/table/*
rm ./ANNTABLE/system5/argu_drug/*
rm ./ANNTABLE/system5/argu_alcohol/*
rm ./ANNTABLE/system5/argu_tobacco/*
rm ./ANNTABLE/system5/argu_emp/*
rm ./ANNTABLE/system5/argu_liv/*
rm ./ANNTABLE/system5/piece_relation/*


conda activate scispacyV5

##############################################################
# Trigger NER - 2-notag Pred
##############################################################        
# yaml_model_name, input_test, output_tes; final-model.pt or best-model.pt
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-trigger-piece.yaml' 'test2_triggers_ner.txt' 'test2_triggers_pred.txt' > ./ner_results/trigger_ner_testsys1.out 2>&1 &

# generate predicted trigger ann (T#) and prediction
# ann_save_dir, conll_order, above_pred, output_pred
python test2ann_events.py './ANNTABLE/system5/ann/' './conll_num/test2_triggers_num.conll' 'test2_triggers_pred.txt' 'test2_triggers_relation.txt'

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


# Subtype Test Data Prepare
python subtype_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt'
python subtype_gt2.py # ground truth


# OUTPUT for classification prediction 
 # 64     input_file0 = './ANNTABLE/system5/piece_relation/test2_triggers_relation.txt' #triggers
 # 65     input_file1 = './ANNTABLE/system5/piece_relation/test2_Drug_relation.txt' #Drug 
 # 66     input_file2 = './ANNTABLE/system5/piece_relation/test2_Alcohol_relation.txt' #Alcohol
 # 67     input_file3 = './ANNTABLE/system5/piece_relation/test2_Tobacco_relation.txt' #Tobacco
 # 68     input_file4 = './ANNTABLE/system5/piece_relation/test2_LivingStatus_relation.txt' #LivingStatus
 # 69     input_file5 = './ANNTABLE/system5/piece_relation/test2_Employment_relation.txt' #Employment
 
# 关系文件 input
# 208     df_med.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_med.csv', header = False) # relative position
# 209     df_emp.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_emp.csv', header = False) # relative position
# 210     df_liv_Status.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_liv_status.csv', header = False) # relative position
# 211     df_liv_Type.to_csv('./ANNTABLE/system5/piece_subtype/subtype_test22_liv_type.csv', header = False) # relative position

# prediction 关系 output 
# ./relation_pred/testsys1-base-pred-subtype-med-123.csv
# ./relation_pred/testsys1-base-pred-subtype-emp-123.csv
# ./relation_pred/testsys1-base-pred-subtype-liv-status-123.csv
# ./relation_pred/testsys1-base-pred-subtype-liv-type-123.csv

python subtype_med_pred.py
python subtype_emp_pred.py
python subtype_livs_pred.py
python subtype_livt_pred.py

##############################################################
# Relation Classification
##############################################################

# Relation Classification Test Data Prepare
python match_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt' 'relation_test22_piece.csv' './relation_pred/test22-piece-trigger-argument-all-poss-relation.txt' 

# Match Prediction
python relation_pcl_pred.py 'relation_test22_piece.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' 'relation_pred/testsys1-match-pred-V123.csv' 'relation_pred/testsys1-match-prob-V123.csv'

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

# ensemble table, input updated relation b/w trigger and argument
python piece_table22.py 'testsys1-argmax-threshold-relation.txt' 'system5'


#./relation_pred/testsys1-base-pred-subtype-med-123.csv
#./ANNTABLE/system5/piece_subtype/subtype_test22_med.csv

# get scotes
python get_results_testsys1.py
vim scoring_testsys1.csv
vim scoring_testsys1_detailed.csv




python add_missing_subtype.py

cp ~/sdoh/Annotations/test_sdoh/mimic/*.txt ~/sdoh/ANNTABLE/system5/table_update/



python get_results_testsys2.py
vim scoring_testsys2.csv
vim scoring_testsys2_detailed.csv


# F1 
OVERALL,OVERALL,OVERALL,3471,3273,2864,0.8750381912618392,0.8251224430999712,0.849347568208778

# Updated F1 
OVERALL,OVERALL,OVERALL,3471,3506,2992,0.8533941814033086,0.8619994237971766,0.8576752185753189


# For Task A
OVERALL,OVERALL,OVERALL,3471,3327,2862,0.8602344454463481,0.8245462402765773,0.8420123565754634

# Bug
4933.ann
2633.ann

2659.ann
2665.ann

2703.ann  # type2





