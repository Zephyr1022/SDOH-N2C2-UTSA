## A Comprehensive Pipeline for Extracting Social Determinants of Health from EHRs

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

Run `./scripts/preprocessing_train.sh`

We get the data under a directory with such setup:

```
Annotation
└── triggers_tag
	├── train
	│   └── ann + txt -> conll -> combined 
 	├── dev
        └── test

```




