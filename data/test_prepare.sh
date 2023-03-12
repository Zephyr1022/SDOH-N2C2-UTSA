##############################################################
#
#
#
#     Generate Test for Trigger with tag
#
#                           
#
##############################################################

# Task C 518
cp ./Annotations/test_sdoh/uw/*.txt ./Annotations/test

# Option I, generate empty ann for triggers in ./Annotations/test (temp)
python test_ann_empty.py

# generate each test trigger ann from empty ann and Re-distribution ann to each trigger folder tag
bash trigger_ner2.sh # test

# copy txt to triggers_tag folders
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Drug/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Alcohol/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Tobacco/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Employment/test/
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/LivingStatus/test/

# generate conll
# convert ann&txt to conll
nohup bash trigger_anntoconll_test.sh > ./anntoconll_results/trigger_tag_test_anntoconll_taskc_dev.out 2>&1 &  

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
#
#
#
#     Generate Train, Dev, and Test for Arguments with tags 
#
#                           
#
##############################################################

# copy txt
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_drug/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_alcohol/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_tobacco/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_liv/test/
cp ./Annotations/test/*.txt ~/sdoh/Annotations/argu_emp/test/

# generate ann and conll for test; argument_extract_all_test.py
nohup bash argument_single_test.sh > ./anntoconll_results/arguments_taskc_dev.out 2>&1 &

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