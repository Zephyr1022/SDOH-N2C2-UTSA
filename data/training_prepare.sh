##############################################################
#
#
#
#                      STEP I - NER (./template)
#
#
#
##############################################################

##############################################################
#
#     Generate Train, Dev for Trigger with tag
#
##############################################################

echo "trigger_types with label = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']"

# copy txt
cp ./Annotations/train/taskc/*.txt ./Annotations/train/temp
cp ./Annotations/dev/taskc/*.txt ./Annotations/dev/temp

# copy ann
cp ./Annotations/train/taskc/*.ann ./Annotations/train/temp
cp ./Annotations/dev/taskc/*.ann ./Annotations/dev/temp

##############################################################
# MIMIC&UW - Temp is a temporary processing folder
##############################################################

# generate trigger ann in folders:  trigger_extract_train.py
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


##############################################################
#
#     Generate Train, Dev for Arguments with tags 
#
##############################################################

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

# generate conll based on and&txt
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

# NER: seperate models for each argument
bash combine_temp_conll.sh

# move argument ner to ./template and conll_num; move_ner_template.sh
bash move_temp_ner_template.sh

# Joint Argument
python generate_train_one.py './NER_ALL/argument_conll/train/' 'train_arg_together_ner.txt' 
bash remover_cc_temp.sh

python generate_dev_test_one.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_ner.txt' # no shuffle
bash remover_cc_temp.sh

# Move ner corpus to template and conlsll_num
mv train_arg_together_ner.txt template
mv dev_arg_together_ner.txt template
mv dev_arg_together_num.conll conll_num


# trigger <><> 
# argument <><><>

# preprocessing data
python process_event_old.py ./template/train_triggers_tag_ner.txt Trigger train_triggers_tag_ner.txt
python process_event_old.py ./template/dev_triggers_tag_ner.txt Trigger dev_triggers_tag_ner.txt

python process_event_old.py ./template/train_arg_together_ner.txt Argument train_arg_together_ner.txt
python process_event_old.py ./template/dev_arg_together_ner.txt Argument dev_arg_together_ner.txt

cat train_triggers_tag_ner.txt space.txt train_arg_together_ner.txt > mimic-uw_train_ner.txt
cat dev_triggers_tag_ner.txt space.txt dev_arg_together_ner.txt > mimic-uw_dev_ner.txt

mv train_triggers_tag_ner.txt ./template
mv train_arg_together_ner.txt ./template
mv dev_triggers_tag_ner.txt ./template
mv dev_arg_together_ner.txt ./template

mv mimic-uw_train_ner.txt ./template
mv mimic-uw_dev_ner.txt ./template



##############################################################
#
#
#
#    STEP II/III Relation Extraction: Classification Samples (./taggers)
#
#
#
##############################################################

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

# remove Redundant documents
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