#!/usr/bin/env python3
##############################################################
#
#
#
#                     Ensemble Model 2
#
#
#
##############################################################
# test11

Part I NER Models (./tagger)
	- Trigger w/ tag: sdoh-26-trigger-tag-uw_train_dev.yaml # trigger with tag, for example <Employment> 
	- Argument Together: sdoh-26-arg-together-uw.yaml

Part II Relation Classification (./model_save)
	- Argument Subtype Models(./model_save)
		- distilbert-subtype-med-baseline-nlp-lr-v2-123.pt # StatusTime: Drug, Alcohol, and Tabacco
		- distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt # StatusEmploy
		- distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt # TypeLiving
		- distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt # StatusTime
		
	- Relation Extraction Model (Match or Not)(./model_save)
		- distilbert-model-match-baseline-nlp-lr-v2-uw.pt # Match, UW and seed = 123

# Trigger NER - 1-tag
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-trigger-tag-uw_train_dev.yaml' 'test1_triggers_ner.txt' 'test1_triggers_pred.txt' > ./ner_results/trigger_ner_test1.out 2>&1 &

# clean ann table
rm ./ANNTABLE/system1/ann/*.ann

# generate predicted trigger ann (T#) in ./ANNTABLE/system1/ann/ and prediction
python test2ann_events.py './ANNTABLE/system1/ann/' './conll_num/test1_triggers_num.conll' 'test1_triggers_pred.txt' 'test1_triggers_relation.txt'

# Argument NER - 1-together <Drug><Type>

# 更改， final-model.pt or best-model.pt
# input_text:best_model, test_data, output_test, 
CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-arg-together-uw.yaml' 'test1_Drug_ner.txt' 'test1_Drug_arguments_pred.txt' > ./ner_results/argument_ner_drug1.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python error_analysis_test.py 'sdoh-26-arg-together-uw.yaml' 'test1_Alcohol_ner.txt' 'test1_Alcohol_arguments_pred.txt' > ./ner_results/argument_ner_alcohol1.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python error_analysis_test.py 'sdoh-26-arg-together-uw.yaml' 'test1_Tobacco_ner.txt' 'test1_Tobacco_arguments_pred.txt' > ./ner_results/argument_ner_tobacco1.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python error_analysis_test.py 'sdoh-26-arg-together-uw.yaml' 'test1_Employment_ner.txt' 'test1_Employment_arguments_pred.txt' > ./ner_results/argument_ner_emp1.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python error_analysis_test.py 'sdoh-26-arg-together-uw.yaml' 'test1_LivingStatus_ner.txt' 'test1_LivingStatus_arguments_pred.txt' > ./ner_results/argument_ner_liv1.out 2>&1 &

# Generate arguments-relation 
python test2ann_arguments.py './conll_num/test1_Drug_num.conll' 'test1_Drug_arguments_pred.txt' 'test1_Drug_relation.txt'
python test2ann_arguments.py './conll_num/test1_Alcohol_num.conll' 'test1_Alcohol_arguments_pred.txt' 'test1_Alcohol_relation.txt'
python test2ann_arguments.py './conll_num/test1_Tobacco_num.conll' 'test1_Tobacco_arguments_pred.txt' 'test1_Tobacco_relation.txt'
python test2ann_arguments.py './conll_num/test1_Employment_num.conll' 'test1_Employment_arguments_pred.txt' 'test1_Employment_relation.txt'
python test2ann_arguments.py './conll_num/test1_LivingStatus_num.conll' 'test1_LivingStatus_arguments_pred.txt' 'test1_LivingStatus_relation.txt'


# Moving trigger and argument to folder 
mv test1_triggers_relation.txt test_relation
mv test1_triggers_pred.txt test_pred

mv test1_Drug_relation.txt test_relation
mv test1_Alcohol_relation.txt test_relation
mv test1_Tobacco_relation.txt test_relation
mv test1_Employment_relation.txt test_relation
mv test1_LivingStatus_relation.txt test_relation

mv test1_Drug_arguments_pred.txt test_pred
mv test1_Alcohol_arguments_pred.txt test_pred
mv test1_Tobacco_arguments_pred.txt test_pred
mv test1_Employment_arguments_pred.txt test_pred
mv test1_LivingStatus_arguments_pred.txt test_pred

#74     input_file0 = './test_relation/test1_triggers_relation.txt' #evernt
#75     input_file1 = './test_relation/test1_Drug_relation.txt' #Drug 
#76     input_file2 = './test_relation/test1_Alcohol_relation.txt' #Alcohol
#77     input_file3 = './test_relation/test1_Tobacco_relation.txt' #Tobacco
#78     input_file4 = './test_relation/test1_LivingStatus_relation.txt' #LivingStatus
#79     input_file5 = './test_relation/test1_Employment_relation.txt' #Employment

# Generate all-poss-relations combinations for 1-tag, 1-together model 
python test11_relation_pred.py './Annotations/test/*.txt' 'relation_test11.csv' 'test11-trigger-argument-all-poss-relation.txt'

# gdtruth_txt, add-token <drug><type> </type><drug> 1406 464 117 115 119
python test11_relation_subtype_pred.py './Annotations/test/*.txt'

#OUTPUT
#'./template_rl/subtype_test11_med.csv', header = False) # relative position
#'./template_rl/subtype_test11_emp.csv', header = False) # relative position
#'./template_rl/subtype_test11_liv_status.csv', header = False) # relative position
#'./template_rl/subtype_test11_liv_type.csv', header = False) # relative position


# Prediction Arguments-Subtype based on test csv above
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './template_rl/subtype_test11_med.csv' './model_save/distilbert-subtype-med-baseline-nlp-lr-v2-123.pt' 'cuda:0' 'relation_pred/test11-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med_test11.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './template_rl/subtype_test11_emp.csv' './model_save/distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt' 'cuda:1' 'relation_pred/test11-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp_test11.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './template_rl/subtype_test11_liv_type.csv' './model_save/distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt' 'cuda:2' 'relation_pred/test11-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type_test11.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './template_rl/subtype_test11_liv_status.csv' './model_save/distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt' 'cuda:3' 'relation_pred/test11-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status_test11.out 2>&1 &


# Match Filter

# get probability
python relation_pcl_pred.py 'relation_test11.csv' './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt' 'cuda:0' 'relation_pred/test11-uw-base-match-pred-V123.csv' 'relation_pred/test11-uw-base-match-prob-V123.csv'

#./template_rl/relation_train.csv 可能需要改
#./template_rl/relation_dev.csv

# option 1: all combinations
'test11-trigger-argument-all-poss-relation.txt'

# option 2: overall threshold
python relation_match.py 'test11-trigger-argument-all-poss-relation.txt' './relation_pred/test11-uw-base-match-prob-V123.csv' 'test11-trigger-argument-thrd-match-relation.txt' 0.1

# option 3: Argmax with threshold
python relation_match_argmax.py 'test11-trigger-argument-all-poss-relation.txt' './relation_pred/test11-uw-base-match-prob-V123.csv' 'test11-trigger-argument-argmax-relation.txt' 0.1

# option 4: Argmax on Status, Filter + threshold + argmax + status
python relation_match_status_argmax.py 'test11-trigger-argument-all-poss-relation.txt' './relation_pred/test11-uw-base-match-prob-V123.csv' 'test11-trigger-argument-status-argmax-relation.txt' 0.1

#OUTPUT:
#- 'test11-trigger-argument-all-poss-relation.txt'
#- 'test11-trigger-argument-thrd-match-relation.txt'
#- 'test11-trigger-argument-argmax-relation.txt'
#- 'test11-trigger-argument-status-argmax-relation.txt'

# Ensemble Event Table
# Fill The Event-Tag Ann: ./Annotations/events_tag/test_table/

conda activate sdohV1

# clean table
rm ~/sdoh/ANNTABLE/system1/table/*.ann
rm ~/sdoh/ANNTABLE/system1/table/*.txt

# groundtruth test txt
cp ~/sdoh/Annotations/test/*.txt ~/sdoh/ANNTABLE/system1/table/  
cp ~/sdoh/ANNTABLE/system1/ann/*.ann ~/sdoh/ANNTABLE/system1/table/

python event_table11.py 'test11-trigger-argument-all-poss-relation.txt' './ANNTABLE/system1/ann/*.ann'

# default setting 
# 146     match_med = pd.read_csv('./template_rl/subtype_test22_med.csv',header = None) # 源文件 
# 147     match_emp = pd.read_csv('./template_rl/subtype_test22_emp.csv',header = None)
# 148     match_liv_status = pd.read_csv('./template_rl/subtype_test22_liv_status.csv',header = None)
# 149     match_liv_type = pd.read_csv('./template_rl/subtype_test22_liv_type.csv',header = None)
# 151     with open("./relation_pred/test11-base-pred-subtype-med-123.csv", 'r') as iFile_pred1: # 预测
# 155     with open("./relation_pred/test11-base-pred-subtype-emp-123.csv", 'r') as iFile_pred2:
# 159     with open("./relation_pred/test11-base-pred-subtype-liv-status-123.csv", 'r') as iFile_pred3:
# 163     with open("./relation_pred/test11-base-pred-subtype-liv-type-123.csv", 'r') as iFile_pred4:

# Get Results, and repeat
python get_results11.py
vim scoring11.csv

# Alternative Optione 
python event_table11.py 'test11-trigger-argument-thrd-match-relation.txt' './ANNTABLE/system1/ann/*.ann' # option 2: overall threshold
python event_table11.py 'test11-trigger-argument-argmax-relation.txt' './ANNTABLE/system1/ann/*.ann' # option 3: Argmax with threshold
python event_table11.py 'test11-trigger-argument-status-argmax-relation.txt' './ANNTABLE/system1/ann/*.ann' # option 4: Argmax on Status