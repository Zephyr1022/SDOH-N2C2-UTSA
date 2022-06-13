#!/bin/bash

# TEST

# generate empty ann for triggers and cp txt in ./Annotations/test
python test_ann_empty.py
# generate conll in ./Annotations/test
bash test_anntoconll2.sh > anntoconll_results/events_test_uw.out 2>&1 &
# combine conlls, test_dir, output_ner, test2 mean no tag trigger 
python generate_event_test.py './Annotations/test/' 'test_events_ner.txt'
bash remover_cc_temp.sh

# clean folder
bash clean_trigger_tag.sh

# generate each trigger ann from empty ann and Re-distribution ann to each trigger folder
bash trigger_ner2.sh

# copy test txt to each trigger folder
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Drug/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Alcohol/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Tobacco/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/Employment/test
cp ./Annotations/test/*.txt ./Annotations/triggers_tag/LivingStatus/test

# generate conll based on ann&txt for test set
bash trigger_anntoconll_test.sh > ./anntoconll_results/trigger_tag_test_anntoconll_us.out 2>&1 & 

# combine conll and add trigger tag
bash combine_trigger_conll.sh

# move conll files to ~./Annotations/triggers_tag_temp ready to combine to one file 
mv test_drug_tag.conll ~/sdoh/Annotations/test/events_tag_1
mv test_alcohol_tag.conll ~/sdoh/Annotations/events_tag_1
mv test_tobacco_tag.conll ~/sdoh/Annotations/events_tag_1
mv test_employment_tag.conll ~/sdoh/Annotations/events_tag_1
mv test_livingstatus_tag.conll ~/sdoh/Annotations/events_tag_1

# clean temp files
bash remover_cc_temp.sh


bash clean_argument_test.sh

# # copy test txt to each folder for convert conll
bash cp_txt_mimic_argument.sh



# generate conll based on and&txt , copy txt, and then combine conll and add tag
bash argument_single_test.sh

# move conll to ~/sdoh/NER/Drug/test # temp
bash mover_ner_test.sh

# Add trigger <Tag> before Argument <Type> in NER and SAVE in ./NER_ALL 
bash run_test.sh 

# Perpare together test1 data ~/sdoh/NER_ALL/Drug 
# combine conll in each NER, generate ner train dataset and save in template
bash argument_one.sh

# Perpare Option 2 separate data and move to template  ~/sdoh/NER/Drug/test
bash argument_separate.sh
bash test2_ner_mv_template.sh
bash test2_num_mv_conll.sh


run_test.sh


python generate_dev_test_one.py './NER_ALL/argument_conll/test/' 'arguments_test_tag_ner.txt'










