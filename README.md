# Extracting-Social-Determinants-of-Health-N2C2

# Extraction of SDOH information from clinical notes

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
- triggers: 
- train: txt ann conll
- dev: txt ann conll
- test: txt
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
	

#!/bin/bash
##############################################################
#
#
#
#                        Description
#
#
#
##############################################################

'''
trigger_types = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
argument_types_med = ['StatusTime','Duration','History','Type','Amount','Frequency','Method']
argument_types_emp = ['StatusEmploy','Duration','History','Type']
argument_types_liv = ['StatusTime','TypeLiving','Duration','History']
'''

'''
extract triggers
- step 1 generate blank ann file based on the groundtruth txt
- step 2 generate single conll from ann&txt (ann2conll)
- step 3 generate combined conll for trigger 
- step 4 ner model prediction
- reset 
- test ann and conll 全是空的
'''