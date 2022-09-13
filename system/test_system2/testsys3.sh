# NER-Trigger -> NER-Argument -> Subtype Classification ->  Relation Classification
# Pred               Pred              Pred                         Fixed   

# reset system5
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


# Test Data Prepare
python match_relation_testsys1.py './Annotations/test_sdoh/mimic/*.txt' 'relation_test22_piece.csv' './relation_pred/test22-piece-trigger-argument-all-poss-relation.txt' 

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


# Match Prediction: ground_truth_relation.txt, testsys1-relation.txt
python relation_cls_gt.py
python cls_pred_gt.py

# ensemble table
python piece_table22.py 'testsys1-relation.txt' 'system5'

# get scotes
python get_results_testsys1.py
vim scoring_testsys1.csv
vim scoring_testsys2.csv


# F1
OVERALL,OVERALL,OVERALL,3471,2877,2720,0.9454292665971498,0.7836358398156151,0.8569628229363578


# The primary criteria for the SDOH challenge will be:

# trigger: "overlap"
# span-only arguments: "exact":  xi,k ≡ xj,l if xi,k matches xj,l exactly; Si,k = (ai,k = T ype; xi,k = [45, 52])
# labeld arguments: "label": span-with-value argument Li,k; label: span not considered, such that xi,k always consider equivalent to xj,l

# StatusTime in A: Li,k = (ai,k = Status; xi,k = [53, 56], si,k = current)
# StatusTime in B: Lj,l = (aj,l = Status; xj,l = [38, 44], sj,l = current)

# Trigger: The primary function of the trigger is to anchor events and disambiguate events. In the social determinant task, the trigger is a mechanism for collecting related information (arguments). The actual text in the trigger spans, for example "Alcohol Use" for an Alcohol event, generally do not add additional information to the event. You are correct that longer predicted trigger spans may improve matching with the gold trigger spans under the "overlap" criteria; however, there are many social history sections with multiple events of the same type. For example, many social history sections contain multiple Tobacco or multiple Drug events. When multiple events of the same type are described in a social history section, the events will generally be described in close proximity, often the same sentence. The scoring routine only allows a predicted event to match with a single gold event (vice versa). If a system always predicts the trigger span for a given event type (e.g. Tobacco) to be the entire document, the system will only be capable of matching to a single gold event of that event type. Additionally, the routine looks for the first overlapping gold and predicted trigger spans for a given event type. If the predicted span is the entire document, the routine may not pair of the predicted trigger with the most relevant gold trigger. Generating slightly longer predicted trigger spans may result in a modest performance gain; however, predicting very long trigger spans (e.g. entire sentences or the entire document) will likely result in a performance drop.

# Labeled arguments: The labeled arguments include label subtypes, for example {none, current, past} for Assertion. These subtype labels capture most of the salient information from the span associated with the labeled arguments. The scoring routine does not consider the span associated with labeled arguments. Rather, it considers the pairing of a trigger span with the subtype label (e.g. none, current, or past for Assertion). 
# - This is similar to treating the trigger prediction as a multi-label classification task. There are many cases in the data where multiple events of the same type are described in close proximity and the labeled arguments have different values. 
# - For example, a sentence may describe both previous and current tobacco use, such that there are two Tobacco triggers, each with an Assertion argument, where one Assertion = past and one Assertion = current. To correctly predict these cases, a system must be able to correctly identify both trigger spans separately.

# At the conclusion of the challenge, the data will be released for academic/research purposes. We anticipate this more general data release to occur in late 2023.

# Subtask A involves training on MIMIC data and evaluating on MIMIC data. 
# Subtask B involves training on MIMIC data and evaluating on University of Washington (UW) data. 
# Subtask C involves training on both MIMIC and UW data and evaluating on UW data.

# Participating teams are allowed to use external data, knowledge sources, etc. to develop their extraction models. We ask that these additional resources are clearly described when documenting the system.

# Subtask A: Extraction

# Subtask A explores in-domain extraction performance by training and evaluating on MIMIC-III data. In Subtask A, the training set consists of notes from MIMIC-III (). This MIMIC-III training set () was released in February. On the Monday of evaluation week, a test set consisting of MIMIC-III notes () will be released, and system outputs for this test set must be submitted by Tuesday of evaluation week.

 

# Subtask B: Generalizability

# Subtask B explores generalizability by training on MIMIC-III data and evaluating on University of Washington (UW) data. Subtask B will use the same MIMIC-III training set as Subtask A (). On the Monday of evaluation week, a test set consisting of UW notes () will be released, and system outputs for this test set must be submitted by Tuesday of evaluation week.

 

# Subtask C: Learning Transfer

# Subtask C explores transfer learning by training on both MIMC-III and UW data and evaluating on UW data. The training set for Subtask C will consist of the MIMIC-III training set from Subtasks A & B () and an additional UW training set (). To allow the generalizability experimentation of Subtask B, the UW training data for Subtask C () will not be released until Wednesday of evaluation week. On the Thursday of evaluation week, a separate test set consisting of UW notes () will be released, and system outputs for this test set must be submitted by Friday of evaluation week.


# Scoring script – I made a minor change to the scoring script related to the presence of tab (‘\t’) characters in the *.ann BRAT files. The updated script can be pip installed using the upgrade flag (see https://github.com/Lybarger/brat_scoring).

# The abstract should describe the extraction methodology, including architecture, external resources, training procedure, and other details associated with your team's approach. We hope all teams that participated in the challenge will submit an abstract, regardless of the achieved performance. We will present a range of methodologies at the workshop, not just the top performing system(s). Please follow the AMIA podium abstract formatting guidelines (https://amia.org/education-events/amia-2022-annual-symposium/calls-participation).


# We will be hosting a workshop this fall to present the results of the n2c2 challenge tracks and allow participants to share their work. The n2c2 workshop will be an all-day, in-person event on Friday, November 4, which is the day before the start of the AMIA Annual Symposium. The workshop will be at the same venue as the AMIA Annual Symposium, specifically the Washington Hilton hotel in Washington, DC. We will send a registration link and agenda for the workshop in September.

# At the workshop, participating teams will present their work through plenary talks and poster sessions. The abstracts submitted by teams describing their methodology will be used to identify plenary talks and poster presentations. Please note that these abstracts will not be published, as they will not be formally peer-reviewed. In conjunction with the Track 2 SDOH extraction challenge, there will be a JAMIA focus issue on SDOH (see: https://academic.oup.com/jamia/pages/cfp-social-determinants). We will be summarizing the Track 2 challenge results and methodologies in the paper through this special focus issue; however, the submitted abstracts will not be automatically published through the JAMIA focus issue on SDOH. Teams are welcome to submit manuscripts to the JAMIA focus issue on SDOH or other venues.

# As always, please let me know if you have any questions or concerns. 


