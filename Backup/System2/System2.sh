# Description:
#  trigger                          argument
#  1-tag      triggers_tag   |      1-together         NER_ALL
#  2-notag    triggers       |      2-seperate         NER

##############################################################
#                     Baseline Model 1
#                               # test22
##############################################################

Part I NER Models (./tagger)
    - trigger: 
        sdoh-26-trigger-uw_train_dev.yaml
        sdoh-26-trigger-tag-uw_train_dev.yaml
        
    - argument: 
        - sdoh-26-drug-uw_train_dev.yaml 
        - sdoh-26-alcohol-uw_train_dev.yaml
        - sdoh-26-tobacco-uw_train_dev.yaml 
        - sdoh-26-employment-uw_train_dev.yaml
        - sdoh-26-livingstatus-uw_train_dev.yaml
        
# Trigger and Arguments 改变了 Subtype 也会相应的改变                    
Part II Relation Classification (./model_save)
    - Argument Subtype Models (Multi-Class)
        - distilbert-subtype-med-baseline-nlp-lr-v2-123.pt # StatusTime: Drug, Alcohol, and Tabacco
        - distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt # StatusEmploy
        - distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt # TypeLiving
        - distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt # StatusTime

    - Relation Extraction Model (Match or Not)
        - distilbert-model-match-baseline-nlp-lr-v2-uw.pt # Match, UW and seed = 123
        
        
# Trigger NER - 2-notag
# yaml_model_name, input_test, output_test  # final-model.pt or best-model.pt

CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-trigger-mimic_train_dev.yaml' 'test2_triggers_ner.txt' 'test2_triggers_pred.txt' > ./ner_results/trigger_ner_mimic_test2.out 2>&1 &

# clean ann table
rm ./ANNTABLE/system2/ann/*.ann

# generate predicted trigger ann (T#) and prediction
# ann_save_dir, conll_order, above_pred, output_pred
# mv test2_triggers_relation.txt test_relation

python test2ann_events.py './ANNTABLE/system2/ann/' './conll_num/test2_triggers_num.conll' 'test2_triggers_pred.txt' 'test2_triggers_relation.txt'


# Argument NER - 2-seperate

# final-model.pt or best-model.pt
# input_text: best_model, test_data, output_test 

CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-drug-mimic_train_dev.yaml' 'test2_Drug_ner.txt' 'test2_Drug_arguments_pred.txt' > ./ner_results/argument_ner_drug.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-26-alcohol-mimic_train_dev.yaml' 'test2_Alcohol_ner.txt' 'test2_Alcohol_arguments_pred.txt' > ./ner_results/argument_ner_alcohol.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-26-tobacco-mimic_train_dev.yaml' 'test2_Tobacco_ner.txt' 'test2_Tobacco_arguments_pred.txt' > ./ner_results/argument_ner_tobacco.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-employment-mimic_train_dev.yaml' 'test2_Employment_ner.txt' 'test2_Employment_arguments_pred.txt' > ./ner_results/argument_ner_employment.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-livingstatus-mimic_train_dev.yaml' 'test2_LivingStatus_ner.txt' 'test2_LivingStatus_arguments_pred.txt' > ./ner_results/argument_ner_livingstatus.out 2>&1 &


# Generate arguments-relation 
python test2ann_arguments.py './conll_num/test2_Drug_num.conll' 'test2_Drug_arguments_pred.txt' 'test2_Drug_relation.txt'
python test2ann_arguments.py './conll_num/test2_Alcohol_num.conll' 'test2_Alcohol_arguments_pred.txt' 'test2_Alcohol_relation.txt'
python test2ann_arguments.py './conll_num/test2_Tobacco_num.conll' 'test2_Tobacco_arguments_pred.txt' 'test2_Tobacco_relation.txt'
python test2ann_arguments.py './conll_num/test2_Employment_num.conll' 'test2_Employment_arguments_pred.txt' 'test2_Employment_relation.txt'
python test2ann_arguments.py './conll_num/test2_LivingStatus_num.conll' 'test2_LivingStatus_arguments_pred.txt' 'test2_LivingStatus_relation.txt'


# Moving trigger and argument to folder 
mv test2_triggers_relation.txt test_relation
mv test2_triggers_pred.txt test_pred

mv test2_Drug_relation.txt test_relation
mv test2_Alcohol_relation.txt test_relation
mv test2_Tobacco_relation.txt test_relation
mv test2_Employment_relation.txt test_relation
mv test2_LivingStatus_relation.txt test_relation

mv test2_Drug_arguments_pred.txt test_pred
mv test2_Alcohol_arguments_pred.txt test_pred
mv test2_Tobacco_arguments_pred.txt test_pred
mv test2_Employment_arguments_pred.txt test_pred
mv test2_LivingStatus_arguments_pred.txt test_pred

# Overlap 发生在何处？？？
# input_file0 = './test_relation/test2_triggers_relation.txt' #Trigger
# input_file1 = './test_relation/test2_Drug_relation.txt' #Drug 
# input_file2 = './test_relation/test2_Alcohol_relation.txt' #Alcohol
# input_file3 = './test_relation/test2_Tobacco_relation.txt' #Tobacco
# input_file4 = './test_relation/test2_LivingStatus_relation.txt' #LivingStatus
# input_file5 = './test_relation/test2_Employment_relation.txt' #Employment

# NEXT STEP 

 # 75     input_file0 = './test_relation/test2_triggers_relation.txt' #triggers
 # 76     input_file1 = './test_relation/test2_Drug_relation.txt' #Drug 
 # 77     input_file2 = './test_relation/test2_Alcohol_relation.txt' #Alcohol
 # 78     input_file3 = './test_relation/test2_Tobacco_relation.txt' #Tobacco
 # 79     input_file4 = './test_relation/test2_LivingStatus_relation.txt' #LivingStatus
 # 80     input_file5 = './test_relation/test2_Employment_relation.txt' #Employment


# Generate all-poss-relations for 2-notag, 2-seperate
# groundtruth_txt, output_csv, all_poss_rel_dir_output, 改文件里的 trigger-argument-relation-pred 文件名

python test22_relation_pred.py './Annotations/test/*.txt' 'relation_test22.csv' 'test22-trigger-argument-all-poss-relation.txt' 

# Generate Arguments-Subtype - 生成 test data output: argument-subtype-all-poss-relation.txt
# gdtruth_txt, add-token <drug><type> text text text </type><drug> 4017 1497 171 413 419

python test22_relation_subtype_pred.py './Annotations/test/*.txt'


# 综上所述 此处得到 match and subtype classification的 test data
# output: relation_test22.csv is  <trigger></trigger> <type> </type>
# output save ./template_rl/subtype_test22_med.csv
# df_med.to_csv('./template_rl/subtype_test22_med.csv', header = False) # relative position
# df_emp.to_csv('./template_rl/subtype_test22_emp.csv', header = False) # relative position
# df_liv_Status.to_csv('./template_rl/subtype_test22_liv_status.csv', header = False) # relative position
# df_liv_Type.to_csv('./template_rl/subtype_test22_liv_type.csv', header = False) # relative position


# Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
# train_data, dev_data, test_data, best_model, device_cuda, result_save

python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_test22_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/test22-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_test22.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_test22_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' 'relation_pred/test22-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_test22.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_test22_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' 'relation_pred/test22-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_test22.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_test22_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' 'relation_pred/test22-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_test22.out 2>&1 &


# Apply Match Filter

# get probability for all poss relationship: relation_test22.csv (<trigger></trigger> <argument></argument>) 遍历
# test_data, best_model, device_cuda, save_test_pred_loc, save_test_prob_loc

python relation_pcl_pred.py 'relation_test22.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/test22-mimic-base-match-pred-V123.csv' 'relation_pred/test22-mimic-base-match-prob-V123.csv'

# ./template_rl/relation_train.csv
# ./template_rl/relation_dev.csv


# Ensemble Event Table

# option 1: all-poss
# input_relation, event_table_dir

python event_table22.py 'test22-trigger-argument-all-poss-relation.txt' './ANNTABLE/system2/ann/*.ann'

# 146     match_med = pd.read_csv('./template_rl/subtype_test22_med.csv',header = None)
# 147     match_emp = pd.read_csv('./template_rl/subtype_test22_emp.csv',header = None)
# 148     match_liv_status = pd.read_csv('./template_rl/subtype_test22_liv_status.csv',header = None)
# 149     match_liv_type = pd.read_csv('./template_rl/subtype_test22_liv_type.csv',header = None)
# 151     with open("./relation_pred/test22-base-pred-subtype-med-123.csv", 'r') as iFile_pred1:
# 155     with open("./relation_pred/test22-base-pred-subtype-emp-123.csv", 'r') as iFile_pred2:
# 159     with open("./relation_pred/test22-base-pred-subtype-liv-status-123.csv", 'r') as iFile_pred3:
# 163     with open("./relation_pred/test22-base-pred-subtype-liv-type-123.csv", 'r') as iFile_pred4:



# option 2: Filter + threshold
# all_poss_data, match_prob, output_match_relation, threshold
python relation_match.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-mimic-base-match-prob-V123.csv' 'test22-trigger-argument-thrd-match-relation.txt' 0.01
python event_table22.py 'test22-trigger-argument-thrd-match-relation.txt' './ANNTABLE/system2/ann/*.ann'


# option 3: Filter + threshold + argmax   
# all_poss_data_relation, relation_match_prob, output_match_relation, threshold
python relation_match_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-mimic-base-match-prob-V123.csv' "test22-trigger-argument-argmax-relation.txt" 0.2
python event_table22.py 'test22-trigger-argument-argmax-relation.txt' './ANNTABLE/system2/ann/*.ann'


# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-mimic-base-match-prob-V123.csv' "test22-trigger-argument-status-argmax-relation.txt" 0.1
python event_table22.py 'test22-trigger-argument-status-argmax-relation.txt' './ANNTABLE/system2/ann/*.ann'



# clean
# ~/sdoh/ANNTABLE/system2/ann/*.ann 之前生成的 trigger ann 
rm ~/sdoh/ANNTABLE/system2/table/*.ann
rm ~/sdoh/ANNTABLE/system2/table/*.txt

# groundtruth test txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system2/table/  
cp ~/sdoh/ANNTABLE/system2/ann/*.ann ~/sdoh/ANNTABLE/system2/table/


# Get Result

conda activate sdohV1
python get_results22.py
vim scoring22.csv

# option 3
OVERALL,OVERALL,OVERALL,8537,8156,7134,0.8746934771947033,0.8356565538245285,0.8547295273467921
OVERALL,OVERALL,OVERALL,3471,3327,2862,0.8602344454463481,0.8245462402765773,0.8420123565754634
OVERALL,OVERALL,OVERALL,5066,4391,3598,0.8194033249829196,0.7102250296091591,0.760917838638046
OVERALL,OVERALL,OVERALL,5066,4391,3599,0.8196310635390571,0.7104224240031584,0.7611293221951992

# clean
# ~/sdoh/ANNTABLE/system2/ann/*.ann 之前生成的 trigger ann 
rm ~/sdoh/ANNTABLE/system2/table/*.ann
rm ~/sdoh/ANNTABLE/system2/table/*.txt
# groundtruth test txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system2/table/  
cp ~/sdoh/ANNTABLE/system2/ann/*.ann ~/sdoh/ANNTABLE/system2/table/

# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-mimic-base-match-prob-V123.csv' "test22-trigger-argument-status-argmax-relation.txt" 0.01
python event_table22.py 'test22-trigger-argument-status-argmax-relation.txt' './ANNTABLE/system2/ann/*.ann'



python relation_match_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-mimic-base-match-prob-V123.csv' "test22-trigger-argument-argmax-relation.txt" 0.01
python event_table22.py 'test22-trigger-argument-argmax-relation.txt' './ANNTABLE/system2/ann/*.ann'


# Get Result
python get_results22.py
vim scoring22.csv















# gold_dir = "./Annotations/test_original/mimic/",
# predict_dir = "./ANNTable/system2/table/",
# output_path = "scoring_bert_uw.csv",

# bert error_analysis_test.py bset_model 改成了 final_model
# CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-84-event.yaml' 'test_trigger_tag_ner.txt' 'test1_bert_trigger_pred.txt' > ./ner_results/trigger_ner_test2_uw.out 2>&1 &
# python test2ann_events.py './Annotations/triggers/test/' './conll_num/test2_triggers_num.conll' 'test2_flair_triggers_pred.txt' 'test2_triggers_relation.txt'
# Trigger OUTPUT: test2_flair_events_pred.txt + ANN Table
# 生成事件表格：'./Annotations/triggers/test/' 这三个model共用一个data, events folder 变量关联 三选一,先
# bert generate ann and prediction
# python test2ann_events.py './ANNTable/system2/ann/' './conll_num/test_trigger_tag_num.conll' 'test1_bert_trigger_pred.txt' 'test1_bert_trigger_relation.txt'
# rm ./Annotations/triggers/test/*.ann
# Argument Prediction 
#CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-84-drug-uw.yaml' 'test2_Drug_ner.txt' 'test2_Drug_arguments_pred.txt' > ./ner_results/argument_ner_test21_uw.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-84-alcohol-uw.yaml' 'test2_Alcohol_ner.txt' 'test2_Alcohol_arguments_pred.txt' > ./ner_results/argument_ner_test22_uw.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-84-tobacco-uw.yaml' 'test2_Tobacco_ner.txt' 'test2_Tobacco_arguments_pred.txt' > ./ner_results/argument_ner_test23_uw.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-84-employment-uw.yaml' 'test2_Employment_ner.txt' 'test2_Employment_arguments_pred.txt' > ./ner_results/argument_ner_test24_uw.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-84-livingstatus-uw.yaml' 'test2_LivingStatus_ner.txt' 'test2_LivingStatus_arguments_pred.txt' > ./ner_results/argument_ner_test25_uw.out 2>&1 &
# python test2_relation_pred.py './Annotations/test/*.txt' 'relation_test22.csv' 'test22-trigger-argument-all-poss-relation.txt' 



# get probability
# test_data, best_model, device_cuda, test_pred_loc, test_prob_loc
python relation_pcl_pred.py 'relation_test22.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' 'cuda:0' 'relation_pred/test22-uw-base-match-pred-V123.csv' 'relation_pred/test22-uw-base-match-prob-V123.csv'

# option 4 #Argmax on Status
python relation_match_status_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-uw-base-match-prob-V123.csv' "test22-trigger-argument-status-argmax-relation.txt" 0.1

# option4: Filter + threshold + argmax + status
python event_table22.py 'test22-trigger-argument-status-argmax-relation.txt' './ANNTable/system3/ann/*.ann' '123'


# option 3 #Argmax   # all_poss_data, match_prob, output_match_relation, threshold
python relation_match_argmax.py 'test22-trigger-argument-all-poss-relation.txt' './relation_pred/test22-base-match-prob-V123.csv' "test22-trigger-argument-argmax-relation.txt" 0.3
# option 3: Filter + threshold + argmax
python event_table22.py 'test22-trigger-argument-argmax-relation.txt' './ANNTable/system2/ann/*.ann' '123' 

python get_results.py

vim scoring_bert_uw.csv

python get_results.py

vim scoring_event_tag.csv

python get_results21.py

vim scoring21.csv