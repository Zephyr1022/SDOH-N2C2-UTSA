# combine conll in each NER, generate ner train dataset and save in template
python generate_train.py sdoh-26-drug.yaml #output: train_argu_drug_ner.txt
bash remover_cc_temp.sh # clean temp file

python generate_argu_train.py sdoh-26-alcohol.yaml #output: train_argu_alcohol_ner.txt
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-tobacco.yaml #output:train_argu_tobacco_ner.txt
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-livingstatus.yaml #output:train_argu_liv_ner.txt
bash remover_cc_temp.sh 

python generate_argu_train.py sdoh-26-employment.yaml #output:train_argu_emp_ner.txt
bash remover_cc_temp.sh 

# dev
python generate_argu_dev.py sdoh-26-drug.yaml #output: dev_argu_drug_ner.txt (training propose) & dev_argu_drug_tag.conll (predict propose)
bash remover_cc_temp.sh 

python generate_argu_dev.py sdoh-26-alcohol.yaml
bash remover_cc_temp.sh

python generate_argu_dev.py sdoh-26-tobacco.yaml
bash remover_cc_temp.sh

python generate_argu_dev.py sdoh-26-livingstatus.yaml
bash remover_cc_temp.sh

python generate_argu_dev.py sdoh-26-employment.yaml
bash remover_cc_temp.sh


