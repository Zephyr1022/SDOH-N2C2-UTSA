## A marker-based neural network system for extracting social determinants of health

Code for Extracting-Social-Determinants-of-Health-N2C2
The prediction task A, B, and C involves identifying trigger and argument spans, normalizing arguments, and predicting links between trigger and argument spans. 

### Environment
- wheel
- pandas
- tqdm
- spacy>=3.0.0 with languange model "en_core_web_sm"

### Datasets

- MIMIC-III
- University of Washington (UW)

- Training and test data for this task will utilize the SHAC annotations, which will be provided using the BRAT standoff format
- Prepare data processed from [BRAT](https://github.com/Lybarger/brat_scoring)


### Preprocessing

Run `./training_prepare.sh` and `./test_prepare.sh`

We get the data under a directory with such setup:

```
# Description:

Triger: <Trigger><Drug>
Argument: <Argument><Drug><Type>

Annotation
└── triggers_tag
	├── train
	│   └── ann + txt -> conll -> combined 
 	├── dev
        └── test
└── arguments_tag
	├── train
	│   └── ann + txt -> conll -> combined 
 	├── dev
        └── test

```

### Train the model 

Run `./model_train.sh`

### synthesize ann structural data

Run `./pipeline.sh`






