# combine conll in each NER, generate ner train dataset and save in template
# train
python generate_argu_train.py sdoh-26-drug.yaml 'train_argu_drug_ner.txt'
bash remover_cc_temp.sh # clean temp file

python generate_argu_train.py sdoh-26-alcohol.yaml 'train_argu_alcohol_ner.txt'
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-tobacco.yaml 'train_argu_tobacco_ner.txt'
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-livingstatus.yaml 'train_argu_liv_ner.txt'
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-employment.yaml 'train_argu_emp_ner.txt'
bash remover_cc_temp.sh 

# dev
#output: dev_argu_drug_ner.txt (training propose) & dev_argu_drug_tag.conll (predict propose)

python generate_argu_dev_test.py sdoh-26-drug.yaml 'dev_argu_drug'
bash remover_cc_temp.sh 

python generate_argu_dev_test.py sdoh-26-alcohol.yaml 'dev_argu_alcohol'
bash remover_cc_temp.sh

python generate_argu_dev_test.py sdoh-26-tobacco.yaml 'dev_argu_tobacco'
bash remover_cc_temp.sh

python generate_argu_dev_test.py sdoh-26-livingstatus.yaml 'dev_argu_liv'
bash remover_cc_temp.sh

python generate_argu_dev_test.py sdoh-26-employment.yaml 'dev_argu_emp'
bash remover_cc_temp.sh


