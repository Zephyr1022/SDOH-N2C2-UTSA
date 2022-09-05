#!/bin/bash
##############################################################
#
#
#
#     Generate Train, Dev, and Test for Trigger with tags 
#
#                            Trigger With Tag Using JupyterLab
#
##############################################################
# test

# clean txt, ann, conll in ./Annotations/triggers/Drug/test/*.txt&conll&ann
bash remove_txt_ann_conll.sh

# reset temp folder  零时文件夹， 换 corpus
rm ./Annotations/test/*

# clean ann txt and conll
rm ./Annotations/triggers_tag/Drug/test/*
rm ./Annotations/triggers_tag/Alcohol/test/*
rm ./Annotations/triggers_tag/Tobacco/test/*
rm ./Annotations/triggers_tag/Employment/test/*
rm ./Annotations/triggers_tag/LivingStatus/test/*

# reset trigger_conll folder 
rm ./Annotations/triggers_tag/trigger_conll/test/*

# reset triggers folder 
rm ./Annotations/triggers/test/*

##############################################################
#                   此处需要改
##############################################################
# copy txt
cp ./Annotations/test_sdoh/mimic/*.txt ./Annotations/test

# Option I
# generate empty ann for triggers in ./Annotations/test
python test_ann_empty.py

# Option II
# copy original ann
# cp ./Annotations/test_sdoh/mimic/*.ann ./Annotations/test


# generate each test trigger ann from empty ann and Re-distribution ann to each trigger folder tag
bash trigger_ner2.sh # test

# copy txt to triggers_tag folders
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Drug/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Alcohol/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Tobacco/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Employment/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/LivingStatus/test/

# generate conll
# convert ann&txt to conll, 6 produces; generate conll based on ann&txt for test set
bash trigger_anntoconll_test.sh > ./anntoconll_results/trigger_tag_test_anntoconll_piece.out 2>&1 &  

# combine conll files and then add trigger tag, "process.py"
bash combine_trigger_conll.sh # test

# merge conll to target folder trigger_conll/train, dev, test, ready to combine to one file 
mv test_drug_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_alcohol_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_tobacco_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_employment_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_livingstatus_tag.conll ./Annotations/triggers_tag/trigger_conll/test

# combimed conll Again for ner training
python generate_dev_test_one.py './Annotations/triggers_tag/trigger_conll/test/' 'test1_triggers_ner.txt' # 'test_triggers_tag_ner.txt'
bash remover_cc_temp.sh 

# move to template: prediction
mv test1_triggers_ner.txt ./template

# conll_num: reference
mv test1_triggers_num.conll ./conll_num

##############################################################
#                      Test HERE
##############################################################
# copy test txt to ./Annotations/test/test1
cp ./Annotations/test/*.txt ./Annotations/triggers/test
cp ./Annotations/test/*.ann ./Annotations/triggers/test

# generate conll in ./Annotations/triggers/test
bash test_anntoconll2.sh > anntoconll_results/triggers_test2.out 2>&1 &

# combine conlls, test_dir, output_ner; test2 mean no tag trigger 
python generate_event_test.py './Annotations/triggers/test/' 'test2_triggers_ner.txt' 
bash remover_cc_temp.sh

mv test2_triggers_ner.txt ./template
mv test2_triggers_num.conll ./conll_num



##############################################################
#
#
#
#     Generate Train, Dev, and Test for Arguments with tags 
#
#                            Trigger With Tag Using JupyterLab
#
##############################################################
# clean arguments data
rm ~/sdoh/Annotations/argu_drug/test/*
rm ~/sdoh/Annotations/argu_alcohol/test/*
rm ~/sdoh/Annotations/argu_tobacco/test/*
rm ~/sdoh/Annotations/argu_liv/test/*
rm ~/sdoh/Annotations/argu_emp/test/*

# Clean NER and NER_ALL
rm ~/sdoh/NER/Alcohol/test/*.conll
rm ~/sdoh/NER/Drug/test/*.conll
rm ~/sdoh/NER/Employment/test/*.conll
rm ~/sdoh/NER/LivingStatus/test/*.conll
rm ~/sdoh/NER/Tobacco/test/*.conll

rm ./NER_ALL/Alcohol/test/*.conll
rm ./NER_ALL/Drug/test/*.conll
rm ./NER_ALL/Employment/test/*.conll
rm ./NER_ALL/LivingStatus/test/*.conll
rm ./NER_ALL/Tobacco/test/*.conll
rm ./NER_ALL/argument_conll/test/*.conll

# copy txt
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_drug/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_alcohol/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_tobacco/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_liv/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_emp/test/

# generate ann and conll for test; argument_extract_all_test.py
bash argument_single_test.sh 

wait 

# move combined conll to ~/sdoh/NER/Drug/
bash mover_ner_test.sh

# Add trigger tag <Drug> from NER before argument tag <Type>, store in NER_ALL
bash run_test.sh

# merge in the one folder: train dev test
cp ./NER_ALL/Alcohol/test/*.conll ./NER_ALL/argument_conll/test
cp ./NER_ALL/Drug/test/*.conll ./NER_ALL/argument_conll/test
cp ./NER_ALL/Tobacco/test/*.conll ./NER_ALL/argument_conll/test
cp ./NER_ALL/Employment/test/*.conll ./NER_ALL/argument_conll/test
cp ./NER_ALL/LivingStatus/test/*.conll ./NER_ALL/argument_conll/test

# Perpare Option 1 together test1: data ~/sdoh/NER_ALL/Drug; bash argument_one.sh
python generate_dev_test_one.py './NER_ALL/Drug/test/' 'test1_Drug_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/Alcohol/test/' 'test1_Alcohol_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/Tobacco/test/' 'test1_Tobacco_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/Employment/test/' 'test1_Employment_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/LivingStatus/test/' 'test1_LivingStatus_ner.txt'
bash remover_cc_temp.sh

# move ner to ./template and resource to ./conll_num 
bash test1_ner_mv_template.sh 
bash test1_num_mv_conll.sh

# Perpare Option 2 separate test2: data and move to template  ~/sdoh/NER/Drug/test
# combine conll in each NER, generate ner train dataset and save in template; bash argument_separate.sh
python generate_dev_test2.py './NER/Drug/test/' 'test2_Drug_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test2.py './NER/Alcohol/test/' 'test2_Alcohol_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test2.py './NER/Tobacco/test/' 'test2_Tobacco_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test2.py './NER/Employment/test/' 'test2_Employment_ner.txt'
bash remover_cc_temp.sh

python generate_dev_test2.py './NER/LivingStatus/test/' 'test2_LivingStatus_ner.txt'
bash remover_cc_temp.sh

# move ner to ./template and resource to ./conll_num 
bash test2_ner_mv_template.sh
bash test2_num_mv_conll.sh

##############################################################
#
#
#
#    Relation Extraction: Classification Samples (./taggers)
#                                          
#                                           见 system 2
#
##############################################################


# template
dev_arg_together_uw_ner.txt  dev_trigger_ner.txt         test1_Tobacco_ner.txt       test2_Tobacco_ner.txt          train_argu_liv_uw_ner.txt
dev_argu_alcohol_uw_ner.txt  dev_triggers_tag_ner.txt    test1_triggers_ner.txt      test2_triggers_ner.txt         train_argu_tobacco_uw_ner.txt
dev_argu_drug_uw_ner.txt     test1_Alcohol_ner.txt       test2_Alcohol_ner.txt       train_arg_together_uw_ner.txt  train_trigger_ner.txt
dev_argu_emp_uw_ner.txt      test1_Drug_ner.txt          test2_Drug_ner.txt          train_argu_alcohol_uw_ner.txt  train_triggers_tag_ner.txt
dev_argu_liv_uw_ner.txt      test1_Employment_ner.txt    test2_Employment_ner.txt    train_argu_drug_uw_ner.txt
dev_argu_tobacco_uw_ner.txt  test1_LivingStatus_ner.txt  test2_LivingStatus_ner.txt  train_argu_emp_uw_ner.txt

# conll_num
dev_arg_together_uw_num.conll  dev_argu_liv_uw.conll       test1_Alcohol_num.conll       test1_Tobacco_num.conll   test2_Employment_num.conll
dev_argu_alcohol_uw.conll      dev_argu_tobacco_uw.conll   test1_Drug_num.conll          test1_triggers_num.conll  test2_LivingStatus_num.conll
dev_argu_drug_uw.conll         dev_trigger_num.conll       test1_Employment_num.conll    test2_Alcohol_num.conll   test2_Tobacco_num.conll
dev_argu_emp_uw.conll          dev_triggers_tag_num.conll  test1_LivingStatus_num.conll  test2_Drug_num.conll      test2_triggers_num.conll

# Ready for the NER Training


##############################################################
#
#
#
#    Relation Extraction: Classification Samples (./taggers)
#                                          
#                                           见 system 2
#
##############################################################

# test_pred
test2_Alcohol_arguments_pred.txt  test2_Employment_arguments_pred.txt    test2_Tobacco_arguments_pred.txt
test2_Drug_arguments_pred.txt     test2_LivingStatus_arguments_pred.txt  test2_triggers_pred.txt

# test_relation$
test2_Alcohol_relation.txt  test2_Employment_relation.txt    test2_Tobacco_relation.txt
test2_Drug_relation.txt     test2_LivingStatus_relation.txt  test2_triggers_relation.txt


# System5 - Piece Test Temp Folder
# ann  argu_alcohol  argu_drug  argu_emp  argu_liv  argu_tobacco  piece_relation  table

rm ./ANNTABLE/system5/ann/*
rm ./ANNTABLE/system5/table/*
rm ./ANNTABLE/system5/argu_drug/*
rm ./ANNTABLE/system5/argu_alcohol/*
rm ./ANNTABLE/system5/argu_tobacco/*
rm ./ANNTABLE/system5/argu_emp/*
rm ./ANNTABLE/system5/argu_liv/*
rm ./ANNTABLE/system5/piece_relation/*


# relation_pred (including eneities)

test11-base-match-pred-V123.csv              test22-base-pred-subtype-emp-123.csv         testsys1-base-pred-subtype-emp-123.csv
test11-base-match-prob-V123.csv              test22-base-pred-subtype-liv-status-123.csv  testsys1-base-pred-subtype-liv-status-123.csv
test11-base-pred-subtype-emp-123.csv         test22-base-pred-subtype-liv-type-123.csv    testsys1-base-pred-subtype-liv-type-123.csv
test11-base-pred-subtype-liv-status-123.csv  test22-base-pred-subtype-med-123.csv         testsys1-base-pred-subtype-med-123.csv
test11-base-pred-subtype-liv-type-123.csv    test22-mimic-base-match-pred-V123.csv        testsys1-match-pred-V123.csv
test11-base-pred-subtype-med-123.csv         test22-mimic-base-match-prob-V123.csv        testsys1-match-prob-V123.csv
test22-base-match-pred-V123.csv              test22-uw-base-match-pred-V123.csv
test22-base-match-prob-V123.csv              test22-uw-base-match-prob-V123.csv