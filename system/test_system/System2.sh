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
        sdoh-26-trigger-piece.yaml
        
    - argument: 
        - sdoh-26-drug-piece.yaml 
        - sdoh-26-alcohol-piece.yaml
        - sdoh-26-tobacco-piece.yaml 
        - sdoh-26-employment-piece.yaml
        - sdoh-26-livingstatus-piece.yaml
                        
Part II Relation Classification (./model_save)
    - Argument Subtype Models (Multi-Class)
        - distilbert-subtype-med-testsys1.pt # StatusTime: Drug, Alcohol, and Tabacco
        - distilbert-subtype-emp-testsys1.pt # StatusEmploy
        - distilbert-subtype-liv-status-testsys1.pt # StatusTime
        - distilbert-subtype-liv-type-testsys1.pt # TypeLiving 

    - Relation Extraction Model (Match or Not)
        - distilbert-model-match-testsys1.pt

         
##############################################################
# Trigger NER - 2-notag
##############################################################        
# yaml_model_name, input_test, output_tes; final-model.pt or best-model.pt
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-trigger-piece.yaml' 'test2_triggers_ner.txt' 'test2_triggers_pred.txt' > ./ner_results/trigger_ner_testsys1.out 2>&1 &

# clean ann table
rm ./ANNTABLE/system6/ann/*.ann

# generate predicted trigger ann (T#) and prediction
# ann_save_dir, conll_order, above_pred, output_pred
python test2ann_events.py './ANNTABLE/system6/ann/' './conll_num/test2_triggers_num.conll' 'test2_triggers_pred.txt' 'test2_triggers_relation.txt'


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

# Moving triggers and arguments prediction to folder 
mv test2_triggers_pred.txt test_pred
mv test2_Drug_arguments_pred.txt test_pred
mv test2_Alcohol_arguments_pred.txt test_pred
mv test2_Tobacco_arguments_pred.txt test_pred
mv test2_Employment_arguments_pred.txt test_pred
mv test2_LivingStatus_arguments_pred.txt test_pred

mv test2_triggers_relation.txt test_relation
mv test2_Drug_relation.txt test_relation
mv test2_Alcohol_relation.txt test_relation
mv test2_Tobacco_relation.txt test_relation
mv test2_Employment_relation.txt test_relation
mv test2_LivingStatus_relation.txt test_relation

# NEXT STEP 
# Overlap 发生在何处？？？
# input_file0 = './test_relation/test2_triggers_relation.txt' #Trigger
# input_file1 = './test_relation/test2_Drug_relation.txt' #Drug 
# input_file2 = './test_relation/test2_Alcohol_relation.txt' #Alcohol
# input_file3 = './test_relation/test2_Tobacco_relation.txt' #Tobacco
# input_file4 = './test_relation/test2_LivingStatus_relation.txt' #LivingStatus
# input_file5 = './test_relation/test2_Employment_relation.txt' #Employment

# 1682 Tobacco 25 40 StatusTime 25 32     smoking
# 1682 Alcohol 42 57 StatusTime 42 57     occasional EtOH
# 0,"SOCIAL HISTORY:  Housed.   <Tobacco>    <StatusTime>  smoking  </StatusTime>   tobacco  </Tobacco> , 
# occasional EtOH, no other substances",match,1682,1682 Tobacco 25 40 StatusTime 25 32   smoking

##############################################################
# Generate all-poss-relations-trigger-argument  
##############################################################
# for 2-notag, 2-seperate; relation_test22.csv 需要保存
# groundtruth_txt:'./Annotations/test/*.txt', output_csv_test_data_mathc, all_poss_rel_dir_output, 改文件里的文件名

python test22_relation_pred.py 'relation_test22_piece.csv' 'test22-piece-trigger-argument-all-poss-relation.txt' 

# Generate Arguments-Subtype test data
# gdtruth_txt内置了, add-token <drug><type> text text text </type><drug> 4017 1497 171 413 419

python test22_relation_subtype_pred.py 


# 综上所述 此处得到 match and subtype classification的 test data
# template_rl 临时处理文件夹
# output: relation_test22.csv is  <trigger></trigger> <type></type>



##############################################################
# Prediction Subtype (One has Mini Batch Issue, TEST_BATCH_SIZE) 
##############################################################
# train_data, dev_data, test_data, best_model, device_cuda, result_save
# train & dev 是固定的 

# output saved From above - test template （零时文件名）
# - ./template_rl/subtype_test22_med.csv # relative position
# - ./template_rl/subtype_test22_emp.csv # relative position
# - ./template_rl/subtype_test22_liv_status.csv # relative position
# - ./template_rl/subtype_test22_liv_type.csv # relative position

# From Training
# './model_save/distilbert-model-match-testsys1.pt' # match
# './model_save/distilbert-subtype-med-testsys1.pt'
# './model_save/distilbert-subtype-emp-testsys1.pt'
# './model_save/distilbert-subtype-liv-status-testsys1.pt'
# './model_save/distilbert-subtype-liv-type-testsys1.pt'

# Model 变更; relation_pred, template_rl 零时处理文件夹

python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_test22_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' 'relation_pred/test22-pred-subtype-med.csv' > ./relation_results/subtype-med-piece.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_test22_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' 'relation_pred/test22-pred-subtype-emp.csv' > ./relation_results/subtype-emp-piece.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_test22_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' 'relation_pred/test22-pred-subtype-liv-status.csv' > ./relation_results/subtype-liv-status-piece.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_test22_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' 'relation_pred/test22-pred-subtype-liv-type.csv' > ./relation_results/subtype-liv-type-piece.out 2>&1 &


##############################################################
#   Apply Match Filter
##############################################################
# get probability for all-poss-relation: based on relation_test22_piece.csv (<trigger></trigger> <argument></argument>) 遍历
# test_data, best_model, device_cuda, save_test_pred_loc, save_test_prob_loc

python relation_pcl_pred.py 'relation_test22_piece.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:0' 'relation_pred/piece-match-pred-V123.csv' 'relation_pred/piece-match-prob-V123.csv'

# Trigger and Arguments 改变了 Subtype 也会相应的改变   

##############################################################
#                  Ensemble Event Table
##############################################################
# subtype_test模版
# match_med = pd.read_csv('./template_rl/subtype_test22_med.csv',header = None)
# match_emp = pd.read_csv('./template_rl/subtype_test22_emp.csv',header = None)
# match_liv_status = pd.read_csv('./template_rl/subtype_test22_liv_status.csv',header = None)
# match_liv_type = pd.read_csv('./template_rl/subtype_test22_liv_type.csv',header = None)
# 依据模版的预测
# with open("./relation_pred/test22-base-pred-subtype-med-123.csv", 'r') as iFile_pred1:
# with open("./relation_pred/test22-base-pred-subtype-emp-123.csv", 'r') as iFile_pred2:
# with open("./relation_pred/test22-base-pred-subtype-liv-status-123.csv", 'r') as iFile_pred3:
# with open("./relation_pred/test22-base-pred-subtype-liv-type-123.csv", 'r') as iFile_pred4:
# ./template_rl & ./relation_pred 内置设置不变


# option 1: all-poss-relation 
# input_relation, event_table_dir_system_saved
'test22-piece-trigger-argument-all-poss-relation.txt'

# option 2: Filter + threshold
# all_poss_data, match_prob, output_match_relation, threshold
python relation_match.py 'test22-piece-trigger-argument-all-poss-relation.txt' './relation_pred/piece-match-prob-V123.csv' 'piece-tr-ag-thrd-match-relation.txt' 0.01

# option 3: Filter + threshold + argmax   
# all_poss_data_relation, relation_match_prob, output_match_relation, threshold
python relation_match_argmax.py 'test22-piece-trigger-argument-all-poss-relation.txt' './relation_pred/piece-match-prob-V123.csv' "piece-tr-ar-argmax-thrd-relation.txt" 0.01

# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test22-piece-trigger-argument-all-poss-relation.txt' './relation_pred/piece-match-prob-V123.csv' "piece-tr-ar-status-argmax-thrd-relation.txt" 0.1


# clean 表格
rm ~/sdoh/ANNTABLE/system6/table/*.ann
rm ~/sdoh/ANNTABLE/system6/table/*.txt
# groundtruth test txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system6/table/  
cp ~/sdoh/ANNTABLE/system6/ann/*.ann ~/sdoh/ANNTABLE/system6/table/


# 开始填表
python event_table22.py 'test22-piece-trigger-argument-all-poss-relation.txt' 'system6'

rm ~/sdoh/ANNTABLE/system6/table/*.ann
rm ~/sdoh/ANNTABLE/system6/table/*.txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system6/table/  # groundtruth test txt
cp ~/sdoh/ANNTABLE/system6/ann/*.ann ~/sdoh/ANNTABLE/system6/table/

python event_table22.py 'piece-tr-ag-thrd-match-relation.txt' 'system6'

rm ~/sdoh/ANNTABLE/system6/table/*.ann
rm ~/sdoh/ANNTABLE/system6/table/*.txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system6/table/   # groundtruth test txt
cp ~/sdoh/ANNTABLE/system6/ann/*.ann ~/sdoh/ANNTABLE/system6/table/

python event_table22.py 'piece-tr-ar-argmax-thrd-relation.txt' 'system6'

rm ~/sdoh/ANNTABLE/system6/table/*.ann
rm ~/sdoh/ANNTABLE/system6/table/*.txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system6/table/   # groundtruth test txt
cp ~/sdoh/ANNTABLE/system6/ann/*.ann ~/sdoh/ANNTABLE/system6/table/

python event_table22.py 'piece-tr-ar-status-argmax-thrd-relation.txt' 'system6'

rm ~/sdoh/ANNTABLE/system6/table/*.ann
rm ~/sdoh/ANNTABLE/system6/table/*.txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system6/table/   # groundtruth test txt
cp ~/sdoh/ANNTABLE/system6/ann/*.ann ~/sdoh/ANNTABLE/system6/table/


# Get Result

conda activate sdohV1
python get_results22.py
vim scoring22.csv

# option 3
OVERALL,OVERALL,OVERALL,8537,8156,7134,0.8746934771947033,0.8356565538245285,0.8547295273467921
OVERALL,OVERALL,OVERALL,3471,3327,2862,0.8602344454463481,0.8245462402765773,0.8420123565754634
OVERALL,OVERALL,OVERALL,5066,4391,3598,0.8194033249829196,0.7102250296091591,0.760917838638046
OVERALL,OVERALL,OVERALL,5066,4391,3599,0.8196310635390571,0.7104224240031584,0.7611293221951992
# The result for the Fixed NER: 
OVERALL,OVERALL,OVERALL,3471,3341,3286,0.9835378629152948,0.9467012388360703,0.9647680563711099

