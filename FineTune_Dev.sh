#!/bin/bash

# para_data
nohup.out                  sdoh-26-arg-together-uw.yaml  sdoh-26-event-tag-dev.yaml     sdoh-26-livingstatus-uw.yaml  sdoh-26-tobacco.yaml
sdoh-26-alcohol-dev.yaml   sdoh-26-drug-dev.yaml         sdoh-26-event-tag-uw.yaml      sdoh-26-livingstatus.yaml     sdoh-40-event.yaml
sdoh-26-alcohol-uw.yaml    sdoh-26-drug-uw.yaml          sdoh-26-event-tag.yaml         sdoh-26-only-arg-dev.yaml     sdoh-84-event-dev.yaml
sdoh-26-alcohol.yaml       sdoh-26-drug.yaml             sdoh-26-event-test.yaml        sdoh-26-only-arg-fast.yaml    sdoh-84-event.yaml
sdoh-26-all-arg-dev.yaml   sdoh-26-employment-dev.yaml   sdoh-26-event-uw-dev.yaml      sdoh-26-only-arg-new.yaml
sdoh-26-all-arg-fast.yaml  sdoh-26-employment-uw.yaml    sdoh-26-event-uw.yaml          sdoh-26-only-arg.yaml
sdoh-26-all-arg.yaml       sdoh-26-employment.yaml       sdoh-26-event.yaml             sdoh-26-tobacco-dev.yaml
sdoh-26-all.yaml           sdoh-26-event-dev.yaml        sdoh-26-livingstatus-dev.yaml  sdoh-26-tobacco-uw.yaml

# template 
dev_arg_together_uw_ner.txt  dev_events_ner.txt          test1_Tobacco_ner.txt       test_events_ner.txt            train_argu_liv_ner.txt
dev_argu_alcohol_ner.txt     dev_trigger_tag_ner.txt     test2_Alcohol_ner.txt       test_events_tag_ner.txt        train_argu_tobacco_ner.txt
dev_argu_drug_ner.txt        test1_Alcohol_ner.txt       test2_Drug_ner.txt          train_arg_together_uw_ner.txt  train_events_ner.txt
dev_argu_emp_ner.txt         test1_Drug_ner.txt          test2_Employment_ner.txt    train_argu_alcohol_ner.txt     train_trigger_tag_ner.txt
dev_argu_liv_ner.txt         test1_Employment_ner.txt    test2_LivingStatus_ner.txt  train_argu_drug_ner.txt
dev_argu_tobacco_ner.txt     test1_LivingStatus_ner.txt  test2_Tobacco_ner.txt       train_argu_emp_ner.txt

# conll_num
dev_arg_together_uw_num.conll  tag_argu_drug_dev.conll     test1_Alcohol_num.conll       test1_Tobacco_num.conll     test2_LivingStatus_num.conll
dev_events_num.conll           tag_argu_emp_dev.conll      test1_Drug_num.conll          test2_Alcohol_num.conll     test2_Tobacco_num.conll
dev_trigger_tag_num.conll      tag_argu_liv_dev.conll      test1_Employment_num.conll    test2_Drug_num.conll        test_events_num.conll
tag_argu_alcohol_dev.conll     tag_argu_tobacco_dev.conll  test1_LivingStatus_num.conll  test2_Employment_num.conll  test_events_tag_num.conll



# Trigger Prediction: no-tag
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py sdoh-26-event-uw.yaml 'dev_trigger_ner.txt' 'dev_flair_trigger_pred.txt' > ./ner_results/trigger_ner_dev.out 2>&1 &

# Generate Trigger ANN and trigger-relation
python test2ann_events.py './ANNTable/ann/' './conll_num/dev_trigger_tag_num.conll' 'dev_flair_trigger_pred.txt' 'dev_flair_trigger_relation.txt'


# Argument Prediction: separate
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-drug-uw.yaml' 'dev_argu_drug_ner.txt' 'dev_Drug_arguments_pred.txt' > ./ner_results/argument_ner_tes-dev-drug.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-26-alcohol-uw.yaml' 'dev_argu_alcohol_ner.txt' 'dev_Alcohol_arguments_pred.txt' > ./ner_results/argument_ner_dev-alc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-26-tobacco-uw.yaml' 'dev_argu_tobacco_ner.txt' 'dev_Tobacco_arguments_pred.txt' > ./ner_results/argument_ner_dev-toba.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-employment-uw.yaml' 'dev_argu_emp_ner.txt' 'dev_Employment_arguments_pred.txt' > ./ner_results/argument_ner_dev-emp.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-livingstatus-uw.yaml' 'dev_argu_liv_ner.txt' 'dev_LivingStatus_arguments_pred.txt' > ./ner_results/argument_ner_dev-liv.out 2>&1 &

# Generate arguments-relation 
python test2ann_arguments.py './conll_num/tag_argu_drug_dev.conll' 'dev_Drug_arguments_pred.txt' 'dev_Drug_relation.txt'
python test2ann_arguments.py './conll_num/tag_argu_alcohol_dev.conll' 'dev_Alcohol_arguments_pred.txt' 'dev_Alcohol_relation.txt'
python test2ann_arguments.py './conll_num/tag_argu_tobacco_dev.conll' 'dev_Tobacco_arguments_pred.txt' 'dev_Tobacco_relation.txt'
python test2ann_arguments.py './conll_num/tag_argu_emp_dev.conll' 'dev_Employment_arguments_pred.txt' 'dev_Employment_relation.txt'
python test2ann_arguments.py './conll_num/tag_argu_liv_dev.conll' 'dev_LivingStatus_arguments_pred.txt' 'dev_LivingStatus_relation.txt'

# Moving trigger and argument to folder 
mv dev_flair_trigger_pred.txt dev_pred 				# trigger
mv dev_Drug_arguments_pred.txt dev_pred
mv dev_Alcohol_arguments_pred.txt dev_pred
mv dev_Tobacco_arguments_pred.txt dev_pred
mv dev_Employment_arguments_pred.txt dev_pred
mv dev_LivingStatus_arguments_pred.txt dev_pred

mv dev_flair_trigger_relation.txt dev_relation
mv dev_Drug_relation.txt dev_relation
mv dev_Alcohol_relation.txt dev_relation
mv dev_Tobacco_relation.txt dev_relation
mv dev_Employment_relation.txt dev_relation
mv dev_LivingStatus_relation.txt dev_relation


# Generate all-poss-relations by combine trigger-relation and argument-relation 

# 2-notag, 2-seperate
python dev22_relation_pred.py './Annotations/dev/mimic/*.txt' 'relation_dev22.csv' 'dev22-trigger-argument-all-poss-relation.txt' 

'''
# groundtruth_txt, output_csv, all_poss_rel_dir_output, 需要改文件里的 trigger-argument-relation-pred 文件名

2724 LivingStatus 16 21 TypeLiving 22 35        with daughter
2724 LivingStatus 16 21 StatusTime 16 21        Lives
1127 Tobacco 31 38 StatusTime 24 30     denies

0759 Alcohol 27 31 ETOH
1090 Alcohol 54 61 alcohol
1090 Alcohol 87 107 prior heavy drinking

0419 StatusTime 16 18 No
0419 Type 30 38 illicits
0430 StatusTime 22 23 +
'''

# Generate Arguments-Subtype Data 
python dev22_relation_subtype_pred.py './Annotations/dev/mimic/*.txt'
'''
# gdtruth_txt, add-token <drug><type> </type><drug> 3409 1197 233 347 377
'''

# Arguments-Subtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_dev22_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/dev22-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_test2.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_dev22_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' 'relation_pred/dev22-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_test2.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_dev22_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' 'relation_pred/dev22-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_test2.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_dev22_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' 'relation_pred/dev22-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_test2.out 2>&1 &

'''
# train_data, dev_data, test_data, best_model, device_cuda, result_save
'''


# Relation Filter

# get probability # pretrained 
python relation_pcl_pred.py 'relation_dev22.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/dev22-base-match-pred-V123.csv' 'relation_pred/dev22-base-match-prob-V123.csv'

'''
# test_data, best_model, device_cuda, test_pred_loc, test_prob_loc
'''

# option 0 all-possible-relation

# option 1 # Threshold
python relation_match.py 'dev22-trigger-argument-all-poss-relation.txt' './relation_pred/dev22-base-match-prob-V123.csv' "dev22-trigger-argument-thrd-match-relation.txt" 0.1

# option 2 # Argmax + Threshold
python relation_match_argmax.py 'dev22-trigger-argument-all-poss-relation.txt' './relation_pred/dev22-base-match-prob-V123.csv' "dev22-trigger-argument-argmax-relation.txt" 0.01

# option 3 #Argmax on Status + Threshold
python relation_match_status_argmax.py 'dev22-trigger-argument-all-poss-relation.txt' './relation_pred/dev22-base-match-prob-V123.csv' "dev22-trigger-argument-status-argmax-relation.txt" 0.01

'''
# all_poss_data, match_prob, output_match_relation, threshold
'''

# Get Results 

# activate environment
conda activate sdohV1

# reset table
rm ./ANNTable/table/*.ann
rm ./ANNTable/table/*.txt
cp ./Annotations/dev/mimic/*.txt ./ANNTable/table/  #groundtruth test txt
cp ./ANNTable/ann/*.ann ./ANNTable/table/


# option0: all-poss
python event_table_dev22.py 'dev22-trigger-argument-all-poss-relation.txt' './ANNTable/ann/*.ann' '123'

# option1: Filter + threshold
python event_table_dev22.py 'dev22-trigger-argument-thrd-match-relation.txt' './ANNTable/ann/*.ann' '123'

# option2: Filter + threshold + argmax
python event_table_dev22.py 'dev22-trigger-argument-argmax-relation.txt' './ANNTable/ann/*.ann' '123' 

# option3: Filter + threshold + argmax + status
python event_table_dev22.py 'dev22-trigger-argument-status-argmax-relation.txt' './ANNTable/ann/*.ann' '123' 



# results
python get_results_dev22.py

# view scores
vim scoring_dev22.csv


#Question: 

#;

#Type Typ2 是否会影响

# 打印一些东西 可以显示进程