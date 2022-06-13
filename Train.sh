## MIMIC SDOH pre-processing

##############################################################
#
#
#
#                  SDOH Event Extraction
#
#
#
##############################################################

# Extract triggers entities, trigger w/o tag	

Annotation
└── triggers_tag
	├── train
	│	└── ann + txt -> conll -> combined 
 	├── dev
    └── test

# reset
rm ./Annotations/triggers/train/*.ann
rm ./Annotations/triggers/train/*.conll
rm ./Annotations/triggers/train/*.txt
rm ./Annotations/triggers/dev/*.ann
rm ./Annotations/triggers/dev/*.conll
rm ./Annotations/triggers/dev/*.txt

# generate ann including triggers
python event_trigger.py sdoh-26-event-uw.yaml train  # train 
python event_trigger.py sdoh-26-event-uw.yaml dev    # dev

## copy txt to ./Annotations/train/mimic/*.txt ./Annotations/events/train 
cp ./Annotations/train/mimic/*.txt ./Annotations/triggers/train
cp ./Annotations/dev/mimic/*.txt ./Annotations/triggers/dev

## generate conll from ann & txt for training and validation
bash event.sh > anntoconll_results/triggers_train_dev.out 2>&1 &

# generate combined conll for trigger 
python generate_event_train.py # train
bash remover_cc_temp.sh # clean temp file
python generate_event_dev.py # dev
bash remover_cc_temp.sh


# Extract triggers entities, trigger w/ tag	
Annotation
└── Triggers_tag
		├── Drug
		│ 	├── train
		│	│	 └── ann + txt -> conll
		│ 	├── dev
		│   └── test
		│
		├── Alcohol
		├── Tobacco
		├── Employment
		├──	LivingStatus
		└── trigger_conll # temp-one

# clean
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

rm ./Annotations/triggers_tag/Drug/test/*
rm ./Annotations/triggers_tag/Alcohol/test/*
rm ./Annotations/triggers_tag/Tobacco/test/*
rm ./Annotations/triggers_tag/Employment/test/*
rm ./Annotations/triggers_tag/LivingStatus/test/*
		
## generate ann
bash trigger_ner.sh

## copy txt to triggers_tag folders
bash cp_txt_mimic_trigger.sh

## convert conll file by ann&txt, 6 produces same file with diff tag
bash trigger_anntoconll.sh > ./anntoconll_results/trigger_anntoconll_together.out 2>&1 # generate conll

## combine conll files and add trigger tag
bash cb_trigger_conll.sh

## merge conll files to one folder trigger_conll/train, dev
bash merge_conll.sh

# combine conll files for ner training 
python generate_train_one.py './Annotations/triggers_tag/trigger_conll/train/' 'train_triggers_tag_ner.txt'  			# train, shuffle
bash remover_cc_temp.sh
python generate_dev_test_one.py './Annotations/triggers_tag/trigger_conll/dev/' 'dev_triggers_tag_ner.txt'			# dev, sort
bash remover_cc_temp.sh



# Extract arguments entities, trigger w/o tag	

Annotation
└── arguments
	├── argu_drug
	│ 	├── train ann + txt -> conll 生成再删除
	│	├── dev
	│	└── test
 	├── argu_alcohol
	├── argu_tobacco
	├── argu_emp
    └── argu_liv

NER
└── Alcohol
	│ 	├── train:  train_drug_Amount.conll, train_drug_Status.conll, ... # <Amount>
	│	├── dev
	│	└── test
 	├── Drug
	├── Tobacco
	├── LivingStatus
    └── Employment

NER_ALL
└── Alcohol
	│ 	├── train: train_drug_Amount_db.conll, train_drug_Status_db.conll. ... # <Drug><Amount>
	│	├── dev
	│	└── test
 	├── Drug
	├── Tobacco
	├── LivingStatus
	├── Employment
    └── argument_conll: all db.conll files


# clean previous training folders 
bash rm_anntxtconll.sh
bash rm_NER.sh # ner and ner_all

# copy txt to ./Annotations/argu_drug/train or dev
bash copy_txt_arg.sh 

# generate ann and conll, combine conll and add argument tag
# option 1 no overlap with status BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# bash argument_single.sh
# option 2 exist overlap JUST TYING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# python argument_extract_all.py sdoh-26-drug.yaml train Type
bash argument_single_overlap.sh

# move combined conll to ~/sdoh/NER/Drug/train and dev, store in NER
bash mover_ner.sh 

# Add trigger tag <Drug> from NER before argument tag <Type>, store in NER_ALL
bash run_train_uw.sh
bash run_dev_uw.sh

# merge conll file to one folder for train dev 
bash merge_conll_argument.sh
 
# generate combimed conll files in one folder for ner training 
# NER seperate 
bash combine_conll.sh 

# NER_ALL together 
python generate_train_one.py './NER_ALL/argument_conll/train/' 'train_arg_together_uw_ner.txt' # shuffle based on filename id
bash remover_cc_temp.sh # clean temp files
python generate_dev_test_one.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_uw_ner.txt'  # train_arg_together_uw_ner.txt
bash remover_cc_temp.sh # clean temp files



## move ner to template 
## mv track conll file to conll_num

mv train_trigger_ner.txt ./template 
mv dev_trigger_ner.txt ./template
mv train_trigger_tag_ner.txt ./template 
mv dev_trigger_tag_ner.txt ./template 
mv train_argu_drug_ner.txt ./template
mv train_argu_alcohol_ner.txt ./template
mv train_argu_tobacco_ner.txt ./template
mv train_argu_liv_ner.txt ./template
mv train_argu_emp_ner.txt ./template
mv dev_argu_alcohol_ner.txt ./template
mv dev_argu_drug_ner.txt ./template
mv dev_argu_tobacco_ner.txt ./template
mv dev_argu_liv_ner.txt ./template
mv dev_argu_emp_ner.txt ./template
mv train_arg_together_uw_ner.txt ./template
mv dev_arg_together_uw_ner.txt ./template

mv dev_trigger_num.conll ./conll_num
mv dev_trigger_tag_num.conll ./conll_num
mv tag_argu_drug_dev.conll ./conll_num
mv tag_argu_alcohol_dev.conll ./conll_num
mv tag_argu_tobacco_dev.conll ./conll_num
mv tag_argu_liv_dev.conll ./conll_num
mv tag_argu_emp_dev.conll ./conll_num
mv dev_arg_together_uw_num.conll ./conll_num



# template 
dev_arg_together_uw_ner.txt  dev_argu_liv_ner.txt      train_arg_together_uw_ner.txt  train_argu_liv_ner.txt
dev_argu_alcohol_ner.txt     dev_argu_tobacco_ner.txt  train_argu_alcohol_ner.txt     train_argu_tobacco_ner.txt
dev_argu_drug_ner.txt        dev_trigger_ner.txt       train_argu_drug_ner.txt        train_trigger_ner.txt
dev_argu_emp_ner.txt         dev_trigger_tag_ner.txt   train_argu_emp_ner.txt         train_trigger_tag_ner.txt