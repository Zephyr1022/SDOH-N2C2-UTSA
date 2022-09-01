# Task A, B, and C
##############################################################
#
#
#
#     Generate Train, Dev, and Test for Trigger with tag
#
#                            Trigger With Tag Using JupyterLab
#
##############################################################
# extract triggers
#	- step 1 generate blank ann file based on the groundtruth txt
#	- step 2 generate single conll from ann&txt (ann2conll)
#	- step 3 generate combined conll for trigger 
#	- step 4 ner model prediction
#	- reset 
#	- test ann and conll 全是空的

echo "trigger_types with label = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']"

# clean training  and dev data 
rm ./template/*
rm ./conll_num/*

# reset temp folder  零时文件夹， 换 corpus
rm ./Annotations/train/temp/*
rm ./Annotations/dev/temp/*

# clean ann txt and conll
rm ./Annotations/triggers_tag/Drug/train/*
rm ./Annotations/triggers_tag/Alcohol/train/*
rm ./Annotations/triggers_tag/Tobacco/train/*
rm ./Annotations/triggers_tag/Employment/train/*
rm ./Annotations/triggers_tag/LivingStatus/train/*

rm ./Annotations/triggers_tag/Drug/dev/*
rm ./Annotations/triggers_tag/Alcohol/dev/*
rm ./Annotations/triggers_tag/Tobacco/dev/*
rm ./Annotations/triggers_tag/Employment/dev/*
rm ./Annotations/triggers_tag/LivingStatus/dev/*

# reset trigger_conll folder 
rm ./Annotations/triggers_tag/trigger_conll/train/*
rm ./Annotations/triggers_tag/trigger_conll/dev/*

# reset triggers folder 
rm ./Annotations/triggers/train/*
rm ./Annotations/triggers/dev/*

##############################################################
#               此处需要更新 - MIMIC&UW
##############################################################
# copy txt
cp ./Annotations/train/mimic/*.txt ./Annotations/train/temp
cp ./Annotations/dev/mimic/*.txt ./Annotations/dev/temp

# copy original ann
cp ./Annotations/train/mimic/*.ann ./Annotations/train/temp
cp ./Annotations/dev/mimic/*.ann ./Annotations/dev/temp
# cp ./Annotations/test/mimic/*.ann ./Annotations/test

# trigger_extract_train.py
# save drug entities T in ./Annotations/triggers_tag/Drug/train

# generate certain trigger ann in folders: 
bash trigger_ner.sh # train and dev

#python trigger_extract_train.py 'Drug' '/train'   # temp
#python event_trigger_tag.py 'Drug' ~/sdoh/Annotations/test '/test'

# copy txt to triggers_tag folders
cp ./Annotations/train/temp/*.txt ./Annotations/triggers_tag/Drug/train/
cp ./Annotations/train/temp/*.txt ./Annotations/triggers_tag/Alcohol/train/
cp ./Annotations/train/temp/*.txt ./Annotations/triggers_tag/Tobacco/train/
cp ./Annotations/train/temp/*.txt ./Annotations/triggers_tag/Employment/train/
cp ./Annotations/train/temp/*.txt ./Annotations/triggers_tag/LivingStatus/train/

cp ./Annotations/dev/temp/*.txt ./Annotations/triggers_tag/Drug/dev/
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers_tag/Alcohol/dev/
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers_tag/Tobacco/dev/
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers_tag/Employment/dev/
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers_tag/LivingStatus/dev/

# 存在overlap 无法正常识别，查看result output debug
# 4567.ann 
#'''
#E4      Drug:T9 Status:T10 Method:T11
#E5      Drug:T9 Status:T12 Method:T13
#'''

echo "Ann2conll Started"

# generate conll
# convert ann&txt to conll, 6 produces
bash trigger_anntoconll.sh > ./anntoconll_results/trigger_anntoconll_mimic.out 2>&1 # train and dev

echo "Ann2conll Finished"

# combine conll files and add trigger tag by "process.py"
bash cb_trigger_conll.sh #train and dev

# merge conll to target folder trigger_conll/train, dev, test, ready to combine to one file 
mv train_trigger_livingstatus.conll ./Annotations/triggers_tag/trigger_conll/train
mv train_trigger_employment.conll ./Annotations/triggers_tag/trigger_conll/train
mv train_trigger_tobacco.conll ./Annotations/triggers_tag/trigger_conll/train
mv train_trigger_alcohol.conll ./Annotations/triggers_tag/trigger_conll/train
mv train_trigger_drug.conll ./Annotations/triggers_tag/trigger_conll/train

mv dev_trigger_livingstatus.conll ./Annotations/triggers_tag/trigger_conll/dev
mv dev_trigger_employment.conll ./Annotations/triggers_tag/trigger_conll/dev
mv dev_trigger_tobacco.conll ./Annotations/triggers_tag/trigger_conll/dev
mv dev_trigger_alcohol.conll ./Annotations/triggers_tag/trigger_conll/dev
mv dev_trigger_drug.conll ./Annotations/triggers_tag/trigger_conll/dev

#For trigger w/o tag
#python generate_event_test.py './Annotations/test/' 'test_trigger_mimic_ner.txt'
# generate_event_test.py ??? 

# combimed conll Again for ner training 
python generate_train_one.py './Annotations/triggers_tag/trigger_conll/train/' 'train_triggers_tag_ner.txt' # shuffle
bash remover_cc_temp.sh # clean temp generated files

python generate_dev_test_one.py './Annotations/triggers_tag/trigger_conll/dev/' 'dev_triggers_tag_ner.txt'
bash remover_cc_temp.sh 

# move to template 
mv train_triggers_tag_ner.txt ./template
mv dev_triggers_tag_ner.txt ./template

# conll_num: reference
mv train_triggers_tag_num.conll ./conll_num
mv dev_triggers_tag_num.conll ./conll_num

# test
# mv test_triggers_tag_ner.txt ./template
# mv test_triggers_tag_num.conll ./conll_num

echo "Ready for Trigger Model Training"

#################################################################
#  extract triggers entities (template, fixed) - trigger w/o tag
################################################################
# No Tag
# generate conll in ./Annotations/test
# bash test_anntoconll2.sh > anntoconll_results/triggers_test_uw.out 2>&1 &
# combine conlls, test_dir, output_ner, test2 mean no tag trigger 
# python generate_event_test.py './Annotations/test/' 'test_trigger_mimic_ner.txt'
# bash remover_cc_temp.sh
# clean triggers_tag/Drug/test folder
# bash clean_trigger_tag.sh

# train and dev
# extract trigger's ann, and save in ./Annotations/triggers/train by argument_file: 'triggers'
python event_trigger.py sdoh-26-trigger.yaml 'train'
python event_trigger.py sdoh-26-trigger.yaml 'dev'

# copy txt to ./Annotations/train/mimic/*.txt ./Annotations/events/train 
cp ./Annotations/train/temp/*.txt ./Annotations/triggers/train
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers/dev

# generate single conll from ann & txt for training
bash event.sh > anntoconll_results/trigger_train_mimic.out 2>&1 &

# generate combined conll for trigger 
python generate_event_train.py 'train_trigger_ner.txt'  #sdoh-26-event.yaml
bash remover_cc_temp.sh

python generate_event_dev.py 'dev_trigger'  #sdoh-26-event-dev.yaml
bash remover_cc_temp.sh

# move ner to template 
mv train_trigger_ner.txt template
mv dev_trigger_ner.txt template
mv dev_trigger_num.conll conll_num

echo "Ready for Trigger NO Tag Model Training"

##############################################################
#
#
#
#     Generate Train, Dev, and Test for Arguments with tags 
#
#                            Trigger With Tag Using JupyterLab
#
##############################################################
# clean all txt,ann,conll in ./Annotations/argu_drug/test, NER and NER_ALL folder
# rm ./test1_relation/*
# bash rm_NER_test.sh

echo "Arguments:"

# clean arguments data
rm ~/sdoh/Annotations/argu_drug/train/*
rm ~/sdoh/Annotations/argu_drug/dev/*

rm ~/sdoh/Annotations/argu_alcohol/train/*
rm ~/sdoh/Annotations/argu_alcohol/dev/*

rm ~/sdoh/Annotations/argu_tobacco/train/*
rm ~/sdoh/Annotations/argu_tobacco/dev/*

rm ~/sdoh/Annotations/argu_liv/train/*
rm ~/sdoh/Annotations/argu_liv/dev/*

rm ~/sdoh/Annotations/argu_emp/train/*
rm ~/sdoh/Annotations/argu_emp/dev/*

# clean NER and NER_ALL folders, # bash rm_NER.sh 
bash rm_NER_train_dev.sh

# copy txt 
cp ./Annotations/train/temp/*.txt ~/sdoh/Annotations/argu_drug/train/
cp ./Annotations/train/temp/*.txt ~/sdoh/Annotations/argu_alcohol/train/
cp ./Annotations/train/temp/*.txt ~/sdoh/Annotations/argu_tobacco/train/
cp ./Annotations/train/temp/*.txt ~/sdoh/Annotations/argu_liv/train/
cp ./Annotations/train/temp/*.txt ~/sdoh/Annotations/argu_emp/train/

cp ./Annotations/dev/temp/*.txt ~/sdoh/Annotations/argu_alcohol/dev/
cp ./Annotations/dev/temp/*.txt ~/sdoh/Annotations/argu_drug/dev/
cp ./Annotations/dev/temp/*.txt ~/sdoh/Annotations/argu_tobacco/dev/
cp ./Annotations/dev/temp/*.txt ~/sdoh/Annotations/argu_liv/dev/
cp ./Annotations/dev/temp/*.txt ~/sdoh/Annotations/argu_emp/dev/

# generate conll based on and&txt , copy txt, and then combine conll and add tag
# option 1 no overlap with status BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# argument_trigger_single.py # deal with overlap
# bash argument_single.sh # generate ann and conll, combine conll and add argument tag
# option 2 exist overlap JUST TYING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# generate ann and conll 
# train and dev
bash argument_single_overlap.sh # argument_extract_all.py # train/temp

# move combined conll to ~/sdoh/NER/Drug/
# train and dev, store in NER
bash mover_ner.sh

# Add trigger tag <Drug> from NER before argument tag <Type>, store in NER_ALL
bash run_train_uw.sh
bash run_dev_uw.sh

# merge in the one folder: train dev test
cp ./NER_ALL/Alcohol/train/*.conll ./NER_ALL/argument_conll/train
cp ./NER_ALL/Drug/train/*.conll ./NER_ALL/argument_conll/train
cp ./NER_ALL/Tobacco/train/*.conll ./NER_ALL/argument_conll/train
cp ./NER_ALL/Employment/train/*.conll ./NER_ALL/argument_conll/train
cp ./NER_ALL/LivingStatus/train/*.conll ./NER_ALL/argument_conll/train

cp ./NER_ALL/Alcohol/dev/*.conll ./NER_ALL/argument_conll/dev
cp ./NER_ALL/Drug/dev/*.conll ./NER_ALL/argument_conll/dev
cp ./NER_ALL/Tobacco/dev/*.conll ./NER_ALL/argument_conll/dev
cp ./NER_ALL/Employment/dev/*.conll ./NER_ALL/argument_conll/dev
cp ./NER_ALL/LivingStatus/dev/*.conll ./NER_ALL/argument_conll/dev

# no tag trigger 
# combine conlls, test_dir, output_ner, test2 mean 
# python generate_event_test.py './Annotations/test/' 'test_trigger_mimic_ner.txt'
# generate_dev_test2.py
# generate_train_tag.py

##############################################################
#
# NER 里只有一个tag
# NER_ALL 里有两个tag, 再combine all the conll under same folder 
#
##############################################################
# Train and dev 

# NER: seperate models for each argument
bash combine_uw_conll.sh #combine_conll.sh # generate_argu_train.py generate_argu_dev_test.py

# move argument ner to ./template and conll_num
bash move_uw_ner_template.sh  # move_ner_template.sh

# NER_ALL: one model for arguments; combine all conll files to one for training; # generate combimed conll for ner training
# shuffle based on filename id
python generate_train_one.py './NER_ALL/argument_conll/train/' 'train_arg_together_uw_ner.txt' 
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_uw_ner.txt' # no shuffle
bash remover_cc_temp.sh

# Move ner corpus to template and conlsll_num
mv train_arg_together_uw_ner.txt template
mv dev_arg_together_uw_ner.txt template
mv dev_arg_together_uw_num.conll conll_num

# combine all 
# python generate_dev_test_one.py './NER_ALL/argument_conll/test/' 'test_arg_together_uw_ner.txt'
# bash remover_cc_temp.sh

##############################################################
#
#
#
#    Relation Extraction: Classification Samples (./taggers)
#
#
#
##############################################################

# Triggers and Arguments 改变了 Subtype 也会相应的改变, 需要重新 predict 
# generate classification groundtruth sample 
# Input, train = match + not_match 
# INPUT: './Annotations/'+ train_dev_test_dir +'/temp/*.ann' 待处理data 暂存在 trian/temp, dev/temp, test

# Match Classification 
# train
python relation_ps.py train # generate positive example  - match      save: relation_train_match.csv
python relation_ne.py train # generate negative example  - not match  save: relation_train_notmatch.csv

# dev
python relation_ps.py dev    #save: relation_dev_match.csv
python relation_ne.py dev    #save: relation_dev_notmatch.csv

# combine positive and negative to train, dev 
# cat ?? 是否会出问题
cat relation_train_match.csv relation_train_notmatch.csv > relation_train.csv # train: relation_train.csv
cat relation_dev_match.csv relation_dev_notmatch.csv > relation_dev.csv       # dev: relation_dev.csv

# Argument Subtype Classification Generate Data Sample: Know distribution(template_rl, train, dev)
python relation_subtype.py train     # train 10933, 3512 981 959 959 
python relation_subtype.py dev       # dev 1177, 416 90 117 117

# move train and dev to template_rl (relation template)
# remove 多余的文件
rm relation_train_match.csv
rm relation_train_notmatch.csv
rm relation_dev_match.csv
rm relation_dev_notmatch.csv

# Ready for Training: Match Filter & subtype classification
mv relation_train.csv template_rl
mv relation_dev.csv template_rl

mv subtype_train_med.csv template_rl
mv subtype_train_emp.csv template_rl
mv subtype_train_liv_status.csv template_rl
mv subtype_train_liv_type.csv template_rl

mv subtype_dev_med.csv template_rl
mv subtype_dev_emp.csv template_rl
mv subtype_dev_liv_status.csv template_rl
mv subtype_dev_liv_type.csv template_rl

# template_rl 的命名很重要， 不同corpus可以存在不同的位置

# Note

# ./template 
# test1_triggers_ner.txt # w/  tag # same to test_triggers_tag_ner.txt
# test2_triggers_ner.txt # w/o tag 

# ./conll_num
# test1_triggers_num.conll  # test_triggers_tag_num.conll
# test2_triggers_num.conll 

# Note 
# xingmeng@aploverseer:~/sdoh/template$ ls 
# dev_trigger_mimic_ner.txt  test1_triggers_ner.txt  train_trigger_mimic_ner.txt
# dev_triggers_tag_ner.txt   test2_triggers_ner.txt  train_triggers_tag_ner.txt

# ALL the train and dev data save in ./template and ./conll_num