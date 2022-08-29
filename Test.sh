#!/bin/bash

# TEST

# Data Preprocessing 

# trigger 1-tag
# clean folder - bash clean_trigger_tag.sh
rm ./Annotations/triggers_tag/Drug/test/*.conll
rm ./Annotations/triggers_tag/Alcohol/test/*.conll
rm ./Annotations/triggers_tag/Tobacco/test/*.conll
rm ./Annotations/triggers_tag/Employment/test/*.conll
rm ./Annotations/triggers_tag/LivingStatus/test/*.conll

rm ./Annotations/triggers_tag/Drug/test/*.txt
rm ./Annotations/triggers_tag/Alcohol/test/*.txt
rm ./Annotations/triggers_tag/Tobacco/test/*.txt
rm ./Annotations/triggers_tag/Employment/test/*.txt
rm ./Annotations/triggers_tag/LivingStatus/test/*.txt

rm ./Annotations/triggers_tag/Drug/test/*.ann
rm ./Annotations/triggers_tag/Alcohol/test/*.ann
rm ./Annotations/triggers_tag/Tobacco/test/*.ann
rm ./Annotations/triggers_tag/Employment/test/*.ann
rm ./Annotations/triggers_tag/LivingStatus/test/*.ann

# generate each trigger ann from empty ann and Re-distribution ann to each trigger folder
bash trigger_ner2.sh

# copy test txt to each trigger folder
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Drug/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Alcohol/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Tobacco/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Employment/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/LivingStatus/test

# generate conll based on ann&txt for test set
bash trigger_anntoconll_test.sh > ./anntoconll_results/trigger_tag_test_anntoconll_uw.out 2>&1 & 

# combine conll and add trigger tag
python process.py ./Annotations/triggers_tag/Drug/test/ Drug test_drug_tag.conll
python process.py ./Annotations/triggers_tag/Alcohol/test/ Alcohol test_alcohol_tag.conll
python process.py ./Annotations/triggers_tag/Tobacco/test/ Tobacco test_tobacco_tag.conll
python process.py ./Annotations/triggers_tag/Employment/test/ Employment test_employment_tag.conll
python process.py ./Annotations/triggers_tag/LivingStatus/test/ LivingStatus test_livingstatus_tag.conll

# move test_drug_tag.conll to ./Annotations/triggers_tag/trigger_conll/test and combine in one file 
mv test_drug_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_alcohol_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_tobacco_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_employment_tag.conll ./Annotations/triggers_tag/trigger_conll/test
mv test_livingstatus_tag.conll ./Annotations/triggers_tag/trigger_conll/test

# combime conll again for ner prediction test_dir output_ner
# test_dir, test_dir +'*.conll', output_ner
python generate_dev_test_one.py './Annotations/triggers_tag/trigger_conll/test/' 'test_trigger_tag_ner.txt'
bash remover_cc_temp.sh



# trigger 2-notag
# reset test txt in ./Annotations/test

# clean txt, ann, conll in ./Annotations/triggers/Drug/test/*.txt&conll&ann
bash remove_txt_ann_conll.sh

# generate empty ann for triggers and save in ./Annotations/test
python test_ann_empty.py

# generate conll in ./Annotations/test
bash test_anntoconll2.sh > anntoconll_results/events_test_uw.out 2>&1 &

# combine conll files, test_dir, output_ner, test2 mean no tag trigger 
python generate_event_test.py './Annotations/test/' 'test_trigger_ner.txt'
bash remover_cc_temp.sh



# Arguments NER Data Preprocessing 

# clean 
bash clean_argument_test.sh

rm ./NER/Drug/test/*.conll
rm ./NER/Alcohol/test/*.conll
rm ./NER/Tobacco/test/*.conll
rm ./NER/LivingStatus/test/*.conll
rm ./NER/Employment/test/*.conll

rm ./NER_ALL/Drug/test/*.conll
rm ./NER_ALL/Alcohol/test/*.conll
rm ./NER_ALL/Tobacco/test/*.conll
rm ./NER_ALL/LivingStatus/test/*.conll
rm ./NER_ALL/Employment/test/*.conll

rm ./Annotations/argu_alcohol/test/*.conll
rm ./Annotations/argu_drug/test/*.conll
rm ./Annotations/argu_tobacco/test/*.conll
rm ./Annotations/argu_emp/test/*.conll
rm ./Annotations/argu_liv/test/*.conll

rm ./Annotations/argu_alcohol/test/*.ann
rm ./Annotations/argu_drug/test/*.ann
rm ./Annotations/argu_tobacco/test/*.ann
rm ./Annotations/argu_emp/test/*.ann
rm ./Annotations/argu_liv/test/*.ann

rm ./Annotations/argu_alcohol/test/*.txt
rm ./Annotations/argu_drug/test/*.txt
rm ./Annotations/argu_tobacco/test/*.txt
rm ./Annotations/argu_emp/test/*.txt
rm ./Annotations/argu_liv/test/*.txt


# copy test txt to each folder for convert conll: Need to Modify HERE!!!!!!
cp ./Annotations/test/*.txt ./Annotations/argu_drug/test
cp ./Annotations/test/*.txt ./Annotations/argu_alcohol/test
cp ./Annotations/test/*.txt ./Annotations/argu_tobacco/test
cp ./Annotations/test/*.txt ./Annotations/argu_emp/test
cp ./Annotations/test/*.txt ./Annotations/argu_liv/test

# generate conll based on and&txt , copy txt, and then combine conll and add tag
bash argument_single_test.sh

# move conll to ~/sdoh/NER/Drug/test # temp
bash mover_ner_test.sh

# Add trigger <Tag> before Argument <Type> in NER and SAVE in ./NER_ALL 
bash run_test.sh 

# Perpare Option 1 together data ~/sdoh/NER_ALL/Drug 
# combine conll in each NER, generate ner train dataset and save in template
# move ner to ./template and resource to ./conll_num 
bash argument_one.sh

# Perpare Option 2 separate data and move to template  ~/sdoh/NER/Drug/test
bash argument_separate.sh





#  Moving 

#bash test1_ner_mv_template.sh 
#bash test1_num_mv_conll.sh
#bash test2_ner_mv_template.sh
#bash test2_num_mv_conll.sh

mv test1_Drug_ner.txt ./template
mv test1_Alcohol_ner.txt ./template
mv test1_Tobacco_ner.txt ./template
mv test1_Employment_ner.txt ./template
mv test1_LivingStatus_ner.txt ./template

mv test1_Drug_num.conll ./conll_num
mv test1_Alcohol_num.conll ./conll_num
mv test1_Tobacco_num.conll ./conll_num
mv test1_Employment_num.conll ./conll_num
mv test1_LivingStatus_num.conll ./conll_num


mv test2_Drug_ner.txt ./template
mv test2_Alcohol_ner.txt ./template
mv test2_Tobacco_ner.txt ./template
mv test2_Employment_ner.txt ./template
mv test2_LivingStatus_ner.txt ./template

mv test2_Drug_num.conll ./conll_num
mv test2_Alcohol_num.conll ~./conll_num
mv test2_Tobacco_num.conll ./conll_num
mv test2_Employment_num.conll ./conll_num
mv test2_LivingStatus_num.conll ./conll_num
