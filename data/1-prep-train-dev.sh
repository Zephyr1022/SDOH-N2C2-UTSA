##############################################################
#
#
#
#     Generate Train, Dev for Trigger with tag
#
#                            
#
##############################################################
# Task A, B, and C

# extract triggers
#	- step 1 generate blank ann file based on the groundtruth txt
#	- step 2 generate single conll from ann&txt (ann2conll)
#	- step 3 generate combined conll for trigger 
#	- step 4 ner model prediction
#	- reset 
#	- test ann and conll

echo "trigger_types with label = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']"

bash clean_train_dev.sh

##############################################################
# MIMIC&UW&ALL - Temp is a temporary processing folder
##############################################################

# task A

# copy txt
cp ./Annotations/train/mimic/*.txt ./Annotations/train/temp
cp ./Annotations/dev/mimic/*.txt ./Annotations/dev/temp
# copy ann
cp ./Annotations/train/mimic/*.ann ./Annotations/train/temp
cp ./Annotations/dev/mimic/*.ann ./Annotations/dev/temp

# task C - conflict with task A

# copy txt
cp ./Annotations/train/all/*.txt ./Annotations/train/temp
cp ./Annotations/dev/all/*.txt ./Annotations/dev/temp
# copy ann
cp ./Annotations/train/all/*.ann ./Annotations/train/temp
cp ./Annotations/dev/all/*.ann ./Annotations/dev/temp

##############################################################
# MIMIC&UW&ALL - Temp is a temporary processing folder
##############################################################

# generate certain trigger ann in folders:  trigger_extract_train.py
bash trigger_ner.sh # train and dev

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

#################################################################
#  extract triggers entities (template, fixed) - trigger w/ tag
################################################################

# generate conll; convert ann & txt to conll, 6 produces
nohup bash trigger_anntoconll.sh > ./anntoconll_results/trigger_anntoconll_tag_all.out 2>&1 & # train and dev

# combine conlls and add add tag by "process.py"
bash cb_trigger_conll.sh # train and dev

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

# combimed conll Again for ner training 
python generate_train_one.py './Annotations/triggers_tag/trigger_conll/train/' 'train_triggers_tag_ner.txt' # shuffle
bash remover_cc_temp.sh # clean temp generated files

python generate_dev_test_one.py './Annotations/triggers_tag/trigger_conll/dev/' 'dev_triggers_tag_ner.txt' # order
bash remover_cc_temp.sh 

# move to template: ner training
mv train_triggers_tag_ner.txt ./template
mv dev_triggers_tag_ner.txt ./template

# conll_num: reference
mv dev_triggers_tag_num.conll ./conll_num


#################################################################
#  extract triggers entities (template, fixed) - trigger w/o tag
################################################################

# extract trigger's ann, and save in ./Annotations/triggers/train by argument_file: 'triggers'
python event_trigger.py sdoh-26-trigger.yaml 'train'
python event_trigger.py sdoh-26-trigger.yaml 'dev'

# copy txt to ./Annotations/train/mimic/*.txt ./Annotations/events/train 
cp ./Annotations/train/temp/*.txt ./Annotations/triggers/train
cp ./Annotations/dev/temp/*.txt ./Annotations/triggers/dev

# generate single conll from ann & txt for training
bash event.sh > ./anntoconll_results/trigger_train_all.out 2>&1 &

# generate combined conll for trigger 
python generate_event_train.py 'train_trigger_ner.txt'  #sdoh-26-event.yaml
bash remover_cc_temp.sh

python generate_event_dev.py 'dev_trigger'  #sdoh-26-event-dev.yaml
bash remover_cc_temp.sh

# task A,B,C
# move ner to template 
mv train_trigger_ner.txt template
mv dev_trigger_ner.txt template
mv dev_trigger_num.conll conll_num




##############################################################
#
#
#
#     Generate Train, Dev, and Test for Arguments with tags 
#
#                            
#
##############################################################

# clean all txt,ann,conll in ./Annotations/argu_drug/test, NER and NER_ALL folder

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

# clean NER and NER_ALL folders; bash rm_NER.sh 
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
# train and dev; argument_extract_all.py # train/temp
# bash argument_single_overlap.sh 

nohup bash argument_single_overlap.sh > ./anntoconll_results/argument_anntoconll_all.out 2>&1 &

# move combined conll to NER
bash mover_ner.sh

# Add trigger tag <Drug> from NER before argument tag <Type>, and then store in NER_ALL
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


##############################################################
#
# NER - only one tag
# NER_ALL - two tags, and then combine all the conll under same folder 
#
##############################################################

# NER: seperate models for each argument
# bash combine_uw_conll.sh
bash combine_temp_conll.sh

# move argument ner to ./template and conll_num; move_ner_template.sh
# bash move_uw_ner_template.sh  
bash move_temp_ner_template.sh

# NER_ALL: one model for arguments; combine all conll files to one for training; generate combimed conll for ner training
# shuffle based on filename id
# python generate_train_one.py './NER_ALL/argument_conll/train/' 'train_arg_together_uw_ner.txt' 
# bash remover_cc_temp.sh
# python generate_dev_test_one.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_uw_ner.txt' # no shuffle
# bash remover_cc_temp.sh

python generate_train_one.py './NER_ALL/argument_conll/train/' 'train_arg_together_ner.txt' 
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_ner.txt' # no shuffle
bash remover_cc_temp.sh

# Move ner corpus to template and conlsll_num
mv train_arg_together_ner.txt template
mv dev_arg_together_ner.txt template
mv dev_arg_together_num.conll conll_num

##############################################################
#                   Ready for Training: NER                  #
##############################################################




##############################################################
#
#
#
#    Relation Extraction: Classification Samples (./taggers)
#
#
#
##############################################################

# IF Triggers and Arguments changed, Subtype will change accordingly, and will need to be re-predicted
# generate classification groundtruth sample 
# Input train = match + not_match 

# Match Classification 

# train ./Annotations/train/temp/
python relation_ps.py train # generate positive example  - match      save: relation_train_match.csv
python relation_ne.py train # generate negative example  - not match  save: relation_train_notmatch.csv

# dev
python relation_ps.py dev    #save: relation_dev_match.csv
python relation_ne.py dev    #save: relation_dev_notmatch.csv

# combine positive and negative to train, dev 
cat relation_train_match.csv relation_train_notmatch.csv > relation_train.csv # train: relation_train.csv
cat relation_dev_match.csv relation_dev_notmatch.csv > relation_dev.csv       # dev: relation_dev.csv

# Argument Subtype Classification Generate Data Sample: Know distribution(template_rl, train, dev)
python relation_subtype.py train     # train 10933, 3512 981 959 959 
python relation_subtype.py dev       # dev 1177, 416 90 117 117

# Move train and dev to template_rl (relation template)
# rm ./template_rl/*.csv # reset

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

# ALL the train and dev data save in ./template and ./conll_num