##############################################################
#
#
#
#                         System 4
#
#
#
##############################################################
# test21: 2-trigger w/o tag, 1-argument together

Part I NER Models (./tagger)
    - Trigger w/o tag: sdoh-26-trigger-tag-uw_train_dev.yaml # trigger with tag, for example <Employment> 
    - Argument together: 
        - sdoh-26-arg-together-uw.yaml


Part II Relation Classification (./model_save) # No Change
    - Argument Subtype Models
        - distilbert-subtype-med-baseline-nlp-lr-v2-123.pt # StatusTime: Drug, Alcohol, and Tabacco
        - distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt # StatusEmploy
        - distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt # TypeLiving
        - distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt # StatusTime
        
    - Relation Extraction Model (Match or Not)
        - distilbert-model-match-baseline-nlp-lr-v2-uw.pt # Match, UW and seed = 123
        

# Ready to prediction

# Generate ALL-POSS-RELATIONS combinations
python test21_relation_pred.py './Annotations/test/*.txt' 'relation_test21.csv' 'test21-trigger-argument-all-poss-relation.txt'

# Generate Arguments-Subtype Prediction
python test21_relation_subtype_pred.py './Annotations/test/*.txt'

# Prediction Arguments-Subtype
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_test21_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/test21-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_test21.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_test21_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' 'relation_pred/test21-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_test21.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_test21_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' 'relation_pred/test21-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_test21.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_test21_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' 'relation_pred/test21-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_test21.out 2>&1 &

# Match Filter
# get probability
python relation_pcl_pred.py 'relation_test21.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' 'cuda:0' 'relation_pred/test21-uw-base-match-pred-V123.csv' 'relation_pred/test21-uw-base-match-prob-V123.csv'

# option 1: all combinations
'test21-trigger-argument-all-poss-relation.txt'

# option 2: overall threshold
python relation_match.py 'test21-trigger-argument-all-poss-relation.txt' './relation_pred/test21-uw-base-match-prob-V123.csv' 'test21-trigger-argument-thrd-match-relation.txt' 0.1

# option 3: Argmax with threshold
python relation_match_argmax.py 'test21-trigger-argument-all-poss-relation.txt' './relation_pred/test21-uw-base-match-prob-V123.csv' 'test21-trigger-argument-argmax-relation.txt' 0.1

# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test21-trigger-argument-all-poss-relation.txt' './relation_pred/test21-uw-base-match-prob-V123.csv' 'test21-trigger-argument-status-argmax-relation.txt' 0.1


# Ensemble Event Table
conda activate sdohV1

# clean table
rm ~/sdoh/ANNTABLE/system4/table/*.ann
rm ~/sdoh/ANNTABLE/system4/table/*.txt

# groundtruth test txt trigger w/ tag saved in system 2
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system4/table/  
cp ~/sdoh/ANNTABLE/system2/ann/*.ann ~/sdoh/ANNTABLE/system4/table/

python event_table21.py 'test21-trigger-argument-all-poss-relation.txt' './ANNTABLE/system4/ann/*.ann'

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
python get_results21.py
vim scoring21.csv

"./Annotations/test_sdoh/all/",
"./ANNTABLE/system4/table/",


# Alternative Optione 
python event_table21.py 'test21-trigger-argument-thrd-match-relation.txt' './ANNTABLE/system4/ann/*.ann' # option 2: overall threshold
python event_table21.py 'test21-trigger-argument-argmax-relation.txt' './ANNTABLE/system4/ann/*.ann' # option 3: Argmax with threshold
python event_table21.py 'test21-trigger-argument-status-argmax-relation.txt' './ANNTABLE/system4/ann/*.ann' # option 4: Argmax on Status
