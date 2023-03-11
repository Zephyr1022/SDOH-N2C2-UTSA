##############################################################
#
#
#
#     Generate Test for Trigger with tag
#
#                           
#
##############################################################
# Task A, B, and C

# clean txt, ann, conll in ./Annotations/triggers/Drug/test/*.txt&conll&ann
bash remove_txt_ann_conll.sh

# reset temp folder
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
#         Here you need to change TaskA or TaskC
##############################################################

# copy Test txt

# Task A 373
cp ./Annotations/test_sdoh/mimic/*.txt ./Annotations/test

# Task B 2010
cp ./Annotations/test_sdoh/taskb/*.txt ./Annotations/test

# Task C 518
cp ./Annotations/test_sdoh/uw/*.txt ./Annotations/test


# Dev 

# Task A 188
cp ./Annotations/dev/mimic/*.txt ./Annotations/test

# Task C 447
cp ./Annotations/dev/all/*.txt ./Annotations/test


# Check
ls -1q *.txt | wc -l


# Option I, generate empty ann for triggers in ./Annotations/test (temp)
python test_ann_empty.py

# Option II, copy original ann
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
nohup bash trigger_anntoconll_test.sh > ./anntoconll_results/trigger_tag_test_anntoconll_taskc_dev.out 2>&1 &  

# combine conll files and then add trigger tag, "process.py"
bash combine_trigger_conll.sh # test

# combine conll files and then add joint trigger tag <Trigger><Drug>, "process.py"
# bash combine_trigger_conll2.sh # test

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

# cp ./Annotations/test_sdoh/mimic/*.txt ./Annotations/test
# cp ./Annotations/test_sdoh/mimic/*.ann ./Annotations/test

# copy test txt to ./Annotations/test/test1
cp ./Annotations/test/*.txt ./Annotations/triggers/test
cp ./Annotations/test/*.ann ./Annotations/triggers/test

# generate conll in ./Annotations/triggers/test
nohup bash test_anntoconll2.sh > ./anntoconll_results/triggers_taska_sys2.out 2>&1 &

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
#                           
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
nohup bash argument_single_test.sh > ./anntoconll_results/arguments_taskc_dev.out 2>&1 &

# generate ann and conll for test; argument_extract_all_test.py + anntoconll2.py
# nohup bash argument_single_test2.sh > ./anntoconll_results/arguments_taska_joint_dev.out 2>&1 &

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