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


NER Models (./tagger)
- trigger: 
	sdoh-26-event-tag.yaml # trigger with tag, for example <Employment> 
	sdoh-26-event.yaml # trigger without tag in data, flair embedding 
	sdoh-84-event.yaml # bert embedding 
	sdoh-40-event.yaml # word embedding 
- argument:
	- Separate: 
		sdoh-26-drug.yaml           # data sample, <Type> 
		sdoh-26-alcohol.yaml
		sdoh-26-tobacco.yaml
		sdoh-26-employment.yaml
		sdoh-26-livingstatus.yaml

	- Together: 
		sdoh-26-only-arg-new.yaml   # data sample, <Employment><Type> 


Argument Subtype Classification Models (./model_save)
# seed = 123 (main)
	- distilbert-subtype-med-baseline-nlp-lr-v2-123.pt # StatusTime: Drug, Alcohol, and Tabacco
	- distilbert-subtype-emp-baseline-nlp-lr-v2-123.pt # StatusEmploy
	- distilbert-subtype-liv-type-baseline-nlp-lr-v2-123.pt # TypeLiving
	- distilbert-subtype-liv-status-baseline-nlp-lr-v2-123.pt # StatusTime
# seed = 0 
	- distilbert-subtype-med-baseline-nlp-lr-v2.pt
	- distilbert-subtype-emp-baseline-nlp-lr-v2.pt
	- distilbert-subtype-liv-type-baseline-nlp-lr-v2.pt
	- distilbert-subtype-liv-status-baseline-nlp-lr-v2.pt


Relation Extraction (Match or Not) (./model_save)
	- distilbert-model-match-baseline-nlp-lr-v2-123.pt (Main)
	
	







It will create the following directory structure:

MIMIC_data
├── test
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── train
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── val
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── flat_features.csv
├── labels.csv
├── timeseries.csv
└── timeserieslab.csv




SDOH_Data
├── Hyperparameter
│   └── para_data
├── NER
│ 	├── taggers
│ 	├── template_rule
│ 	├── conll_num
│   └── template
│ 		├── Trigger
│ 		│	├── trigger_ner.txt
│    	│	└── trigger_tag_ner.txt 
│		│
│   	└── Argument
│			├── Drug_ner.txt
│   		├── Tobacco_ner.txt
│    		├── Alcohol_ner.txt
│    		├── Employment_ner.txt
│    		├── LivingStatus_ner.txt
│    		└── arg_together_uw_ner.txt
│ 
├── Argument Subtype
│   ├── model_save
│   │
│   └── template_rl
│		├── med.csv 
│   	├── emp.csv 
│    	├── liv_status.csv 
│    	└── liv_type.csv 
│ 
├── Match Filter
│   ├── ./model_save/... 
│   ├── relation_train.csv
│   ├── relation_dev.csv
│   └── relation_test.csv
│
└── Ensemble
	  └── relation_pred


Annotations folder
- train: groundtruth
- dev: groundtruth
- test

#Two Type of trigger ner data
- events: 
- train: txt ann conll
- dev: txt ann conll
- test
- ann: predicted trigger ann
- ann_table: events ann & txt

- events_tag: 
- Alcohol
- Drug
- Employment
- LivingStatus
- Tobacco
- train 		# only save ann related with tobacco trigger, txt and generate conll
- dev 
- test
- ann 				#save predicted trigger eneities T
- ann_table 		#save events based on the triggers T in ann 

#Two Type of arguments ner data
- Arguments Temporary Dir
- argu_drug
- argu_alcohol
- argu_tobacco
- argu_liv 
- argu_emp
- train 		# temp folder, use and delete, only save ann related with tobacco trigger, txt and generate conll
- dev 
- test

- Argument Separate
- NER					#<argument-tag>
- Arguments Together
- NER_ALL				#<trigger-tag><argument-ta>



# Extract triggers entities, trigger w/o tag	
Annotation
└── triggers
	├── train
	│	└── ann + txt -> conll
 	├── dev
    └── test
	
## generate ann including triggers
python event_trigger.py sdoh-26-event-uw.yaml train  # train 
python event_trigger.py sdoh-26-event-uw.yaml dev    # dev

## copy txt to ./Annotations/train/mimic/*.txt ./Annotations/events/train 
cp ./Annotations/train/mimic/*.txt ./Annotations/triggers/train
cp ./Annotations/dev/mimic/*.txt ./Annotations/triggers/dev

## generate conll from ann & txt for training
bash event.sh > anntoconll_results/triggers_train_dev.out 2>&1 &

# Extract triggers entities, trigger w/ tag	

Annotation
└── triggers_tag
	├── train
	│	└── ann + txt -> conll
 	├── dev
    └── test

## generate ann
bash trigger_ner.sh

## copy txt to triggers_tag folders
bash cp_txt_mimic_trigger.sh

## convert conll file by ann&txt, 6 produces
bash trigger_anntoconll.sh > ./anntoconll_results/trigger_anntoconll_together.out 2>&1 # generate conll

## combine conll files and add trigger tag
bash cb_trigger_conll.sh

## merge conll files to one folder trigger_conll/train, dev
bash merge_conll.sh

# combine conll files for ner training 
python generate_train_tag.py './Annotations/triggers_tag/trigger_conll/train/' 'train_trigger_tag_ner.txt'  # train
python generate_test.py './Annotations/triggers_tag/trigger_conll/dev/' 'dev_trigger_tag_ner.txt'			# dev


# Extract triggers entities, trigger w/o tag	

Annotation
└── triggers
	├── argu_drug
	│ 	├── train
	│	├── dev
	│	└── test
 	├── argu_alcohol
	├── argu_tobacco
	├── argu_emp
    └── argu_liv

NER
└── Alcohol
	│ 	├── train
	│	├── dev
	│	└── test
 	├── Drug
	├── Tobacco
	├── LivingStatus
    └── Employment

NER_ALL
└── Alcohol
	│ 	├── train
	│	├── dev
	│	└── test
 	├── Drug
	├── Tobacco
	├── LivingStatus
	├── Employment
    └── argument_conll

# clean previous training folders 
bash rm_anntxtconll.sh
bash rm_NER.sh

# copy txt to ./Annotations/argu_drug/train or dev
bash copy_txt_arg.sh 

# generate ann and conll, combine conll and add argument tag

# option 1 no overlap with status BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bash argument_single.sh

# option 2 exist overlap JUST TYING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bash argument_single_overlap.sh

# move combined conll to ~/sdoh/NER/Drug/train and dev, store in NER
bash mover_ner.sh 

# Add trigger tag <Drug> from NER before argument tag <Type>, store in NER_ALL
run_train_uw.sh
run_dev_uw.sh
run_test.sh

# merge conll file to one folder for train dev 
bash merge_conll_argument.sh
 
# generate combimed conll files in one folder for ner training 
python generate_train_tag.py './NER_ALL/argument_conll/train/' 'train_arg_together_uw_ner.txt' # shuffle based on filename id
python generate_test.py './NER_ALL/argument_conll/dev/' 'dev_arg_together_uw_ner.txt'  # train_arg_together_uw_ner.txt
python generate_test.py './NER_ALL/argument_conll/test/' 'arguments_test_tag_ner.txt'

bash remover_cc_temp.sh # clean temp files


## move ner to template 
## mv track conll file to conll_num

# trigger ner model training

# option 1-tag trigger
CUDA_VISIBLE_DEVICES=0 nohup python sdoh_trainer.py sdoh-26-event-tag-uw.yaml > ./ner_results/trigger_tag_ner_train_uw.out 2>&1 &

# option 2-notag trigger
CUDA_VISIBLE_DEVICES=1 nohup python sdoh_trainer.py sdoh-26-event-uw.yaml > ./ner_results/trigger_ner_train_uw.out 2>&1 &


# Evaluation
	