# 1-tag, 1-together
# 2-notag, 2-seperate

##############################################################
#
#
#
#                         System 3
#
#
#
##############################################################
# test12: 1-trigger w/ tag, 2-argument seperate

Part I NER Models (./tagger)
    - Trigger w/ tag: sdoh-26-trigger-tag-uw_train_dev.yaml # trigger with tag, for example <Employment> 
    - Argument: 
        - sdoh-26-drug-uw_train_dev.yaml 
        - sdoh-26-alcohol-uw_train_dev.yaml
        - sdoh-26-tobacco-uw_train_dev.yaml 
        - sdoh-26-employment-uw_train_dev.yaml
        - sdoh-26-livingstatus-uw_train_dev.yaml

Part II Relation Classification (./model_save) # No Change
    - Argument Subtype Models
        - distilbert-subtype-med-baseline-nlp-lr-v2-123.pt # StatusTime: Drug, Alcohol, and Tabacco
        - distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt # StatusEmploy
        - distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt # TypeLiving
        - distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt # StatusTime
        
    - Relation Extraction Model (Match or Not)
        - distilbert-model-match-baseline-nlp-lr-v2-uw.pt # Match, UW and seed = 123
        
        
# Necessary Test Data
# template
test1_Tobacco_ner.txt       test2_Tobacco_ner.txt  
test1_triggers_ner.txt      test2_triggers_ner.txt      
test1_Alcohol_ner.txt       test2_Alcohol_ner.txt     
test1_Drug_ner.txt          test2_Drug_ner.txt         
test1_Employment_ner.txt    test2_Employment_ner.txt   
test1_LivingStatus_ner.txt  test2_LivingStatus_ner.txt 

# conll_num
test1_Employment_num.conll    test2_Drug_num.conll          
test1_LivingStatus_num.conll  test2_Employment_num.conll
test1_Tobacco_num.conll       test2_LivingStatus_num.conll
test1_Alcohol_num.conll       test2_Tobacco_num.conll
test1_Drug_num.conll          test2_Alcohol_num.conll      
test1_triggers_num.conll      test2_triggers_num.conll

#test_relation 临时文件夹 每次换 corpus 都会改变
test1_triggers_relation.txt  test1_Alcohol_relation.txt  test1_Employment_relation.txt
test1_Drug_relation.txt      test1_Tobacco_relation.txt  test1_LivingStatus_relation.txt
test2_triggers_relation.txt  test2_Alcohol_relation.txt  test2_Employment_relation.txt
test2_Drug_relation.txt      test2_Tobacco_relation.txt  test2_LivingStatus_relation.txt

# test_pred
test1_Alcohol_arguments_pred.txt  test1_Employment_arguments_pred.txt    test1_Tobacco_arguments_pred.txt
test1_Drug_arguments_pred.txt     test1_LivingStatus_arguments_pred.txt  test1_triggers_pred.txt
test2_Alcohol_arguments_pred.txt  test2_Employment_arguments_pred.txt    test2_Tobacco_arguments_pred.txt
test2_Drug_arguments_pred.txt     test2_LivingStatus_arguments_pred.txt  test2_triggers_pred.txt


# template_rl 会不断增加
relation_dev.csv            subtype_dev_liv_type.csv       subtype_test11_liv_type.csv    subtype_test22_liv_type.csv   subtype_train_liv_type.csv
relation_train.csv          subtype_dev_med.csv            subtype_test11_med.csv         subtype_test22_med.csv        subtype_train_med.csv
subtype_dev_emp.csv         subtype_test11_emp.csv         subtype_test22_emp.csv         subtype_train_emp.csv
subtype_dev_liv_status.csv  subtype_test11_liv_status.csv  subtype_test22_liv_status.csv  subtype_train_liv_status.csv




# Generate ALL-POSS-RELATIONS combinations
python test12_relation_pred.py './Annotations/test/*.txt' 'relation_test12.csv' 'test12-trigger-argument-all-poss-relation.txt'

# Generate Arguments-Subtype Prediction
python test12_relation_subtype_pred.py './Annotations/test/*.txt'

# Prediction Arguments-Subtype
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_test12_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/test12-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_test12.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_test12_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' 'relation_pred/test12-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_test12.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_test12_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' 'relation_pred/test12-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_test12.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_test12_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' 'relation_pred/test12-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_test12.out 2>&1 &


# Match Filter
# get probability
python relation_pcl_pred.py 'relation_test12.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' 'cuda:0' 'relation_pred/test12-uw-base-match-pred-V123.csv' 'relation_pred/test12-uw-base-match-prob-V123.csv'

# option 1: all combinations
'test12-trigger-argument-all-poss-relation.txt'

# option 2: overall threshold
python relation_match.py 'test12-trigger-argument-all-poss-relation.txt' './relation_pred/test12-uw-base-match-prob-V123.csv' 'test12-trigger-argument-thrd-match-relation.txt' 0.1

# option 3: Argmax with threshold
python relation_match_argmax.py 'test12-trigger-argument-all-poss-relation.txt' './relation_pred/test12-uw-base-match-prob-V123.csv' 'test12-trigger-argument-argmax-relation.txt' 0.1

# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test12-trigger-argument-all-poss-relation.txt' './relation_pred/test12-uw-base-match-prob-V123.csv' 'test12-trigger-argument-status-argmax-relation.txt' 0.1


# Ensemble Event Table

conda activate sdohV1

# clean table
rm ~/sdoh/ANNTABLE/system3/table/*.ann
rm ~/sdoh/ANNTABLE/system3/table/*.txt

# groundtruth test txt trigger w/ tag saved in system 1
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system3/table/  
cp ~/sdoh/ANNTABLE/system1/ann/*.ann ~/sdoh/ANNTABLE/system3/table/

python event_table12.py 'test12-trigger-argument-all-poss-relation.txt' './ANNTABLE/system3/ann/*.ann'

# default setting 
# 146     match_med = pd.read_csv('./template_rl/subtype_test12_med.csv',header = None) # 源文件 
# 147     match_emp = pd.read_csv('./template_rl/subtype_test12_emp.csv',header = None)
# 148     match_liv_status = pd.read_csv('./template_rl/subtype_test12_liv_status.csv',header = None)
# 149     match_liv_type = pd.read_csv('./template_rl/subtype_test12_liv_type.csv',header = None)
# 151     with open("./relation_pred/test12-base-pred-subtype-med-123.csv", 'r') as iFile_pred1: # 预测
# 155     with open("./relation_pred/test12-base-pred-subtype-emp-123.csv", 'r') as iFile_pred2:
# 159     with open("./relation_pred/test12-base-pred-subtype-liv-status-123.csv", 'r') as iFile_pred3:
# 163     with open("./relation_pred/test12-base-pred-subtype-liv-type-123.csv", 'r') as iFile_pred4:

# Get Results, and repeat
python get_results12.py
vim scoring12.csv

"./Annotations/test_sdoh/all/",
"./ANNTABLE/system3/table/",


# Alternative Optione 
python event_table12.py 'test12-trigger-argument-thrd-match-relation.txt' './ANNTABLE/system3/ann/*.ann' # option 2: overall threshold
python event_table12.py 'test12-trigger-argument-argmax-relation.txt' './ANNTABLE/system3/ann/*.ann' # option 3: Argmax with threshold
python event_table12.py 'test12-trigger-argument-status-argmax-relation.txt' './ANNTABLE/system3/ann/*.ann' # option 4: Argmax on Status