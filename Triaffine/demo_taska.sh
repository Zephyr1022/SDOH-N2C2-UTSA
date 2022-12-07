CUDA_VISIBLE_DEVICES=0 python main.py --version sdoh --model SpanAttModelV3 \
--bert_name_or_path ../pretraining-models/bert-large-cased/  \
--learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 72 --train_epoch 50 --score tri_affine \
--truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  \
--bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 \
--share_parser --subword_aggr max --init_std 1e-2 --dp 0.1

CUDA_VISIBLE_DEVICES=0 python main.py --version ace04 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/albert-xxlarge-v2/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 10 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1


CUDA_LAUNCH_BLOCKING=1 python main.py --version sdoh --model SpanAttModelV3 --bert_name_or_path ./pretraining-models/bert-base-uncased/  \
--learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 72 --train_epoch 50 --score tri_affine  \
--truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  \
--bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30  \
--share_parser --subword_aggr max --init_std 1e-2 --dp 0.1


# genia + biobert
CUDA_VISIBLE_DEVICES=0 python main.py --version genia91 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/biobert_v1.1/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 15 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0  --att_dim 320 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.2

# ace05 + bert
CUDA_VISIBLE_DEVICES=0 python main.py --version ace05 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 72 --train_epoch 50 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1

# ace05 + albert
CUDA_VISIBLE_DEVICES=0 python main.py --version ace05 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/albert-xxlarge-v2/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 10 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1


# ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus','StatusTime','StatusEmploy','TypeLiving', 'Type', 'Method', 'History', 'Duration', 'Frequency', 'Amount']

conda create -n GALACTICA python=3.8
conda activate GALACTICA


CUDA_LAUNCH_BLOCKING=1 python sdoh_paper.py


conda create -n triaffine python=3.7
conda activate triaffine
cd ./sdoh/triaffine
 
mkdir -p pretraining-models/bert-base-uncased
wget -P pretraining-models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget -P pretraining-models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -P pretraining-models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/config.json

mkdir -p pretraining-models/roberta-base
wget -P pretraining-models/roberta-base https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
wget -P pretraining-models/roberta-base https://huggingface.co/roberta-base/resolve/main/merges.txt
wget -P pretraining-models/roberta-base https://huggingface.co/roberta-base/resolve/main/vocab.json
wget -P pretraining-models/roberta-base https://huggingface.co/roberta-base/resolve/main/config.json

mkdir -p pretraining-models/bert-large-cased
wget -P pretraining-models/bert-large-cased https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin
wget -P pretraining-models/bert-large-cased https://huggingface.co/bert-large-cased/resolve/main/vocab.txt
wget -P pretraining-models/bert-large-cased https://huggingface.co/bert-large-cased/resolve/main/config.json


mkdir -p pretraining-models/biobert_pubmed
wget -P pretraining-models/biobert_pubmed http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_large_v1.1_pubmed.tar.gz


mkdir -p pretraining-models/Bio_ClinicalBERT
wget -P pretraining-models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/pytorch_model.bin
wget -P pretraining-models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/vocab.txt
wget -P pretraining-models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json

'Bio_ClinicalBERT':'clinicalbert',
'bert-base-uncased': 'base_uncased',


wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1

tar -xf bert_pretrain_output_all_notes_150000.tar.gz

wget -P pretraining-models https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
wget -P pretraining-models https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz


gunzip biobert_large_v1.1_pubmed.tar.gz
gunzip biobert_large_v1.1_pubmed.tar



########################################################################################################################################

CUDA_VISIBLE_DEVICES=0 python main.py --version ace04 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ 
--learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 50 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1

CUDA_VISIBLE_DEVICES=0 python main.py --version genia91 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/biobert_v1.1/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 15 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0  --att_dim 320 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.2

CUDA_VISIBLE_DEVICES=0 python main.py --version genia91 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/biobert_v1.1/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 15 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0  --att_dim 320 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.2

CUDA_VISIBLE_DEVICES=0 python main.py --version genia91 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/biobert_v1.1/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 15 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0  --att_dim 320 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.2

########################################################################################################################################


cd ./sdoh/triaffine
conda activate triaffine

# run python word_embed.py to generate required json files. You need to change the path of word embedding.
# train_parser.py 调用 word_embed

# Vocab count: 8576
# Char vocab count: 103
# POS vocab count: 45

# change 2 处 in  data_util.py 
# if version.find('sdoh') >= 0:
#     type_list = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus','StatusTime','StatusEmploy','TypeLiving', 'Type', 'Method', 'History', 'Duration', 'Frequency', 'Amount']


python word_embed.py



CUDA_LAUNCH_BLOCKING=1 nohup python main.py --version sdoh --model SpanAttModelV3 --bert_name_or_path ./pretraining-models/Bio_ClinicalBERT   \
--learning_rate 5e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 22 --score tri_affine   \
--truncate_length 256 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0   \
--att_dim 256 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30   \
--share_parser --subword_aggr max --init_std 1e-2 --dp 0.2 > triaffine_taskc_trigger.out 2>&1 &

# nohup
# > triaffine_taska_trigger.out 2>&1 &


# Task A 
Best_Dev_Epoch27 {'p': 0.8198529411764706, 'r': 0.7957181088314005, 'f1': 0.8076052512449071}
Best_Test_Epoch27 {'p': 0.818137493905412, 'r': 0.7979077508321446, 'f1': 0.8078960038517092}

# Task B
Best_Dev_Epoch15 {'p': 0.8334494773519163, 'r': 0.6857798165137615, 'f1': 0.7524378735451401}
Best_Test_Epoch15 {'p': 0.756516310285625, 'r': 0.6072920147009689, 'f1': 0.6737402988532375}

# Task C
Best_Dev_Epoch14 {'p': 0.8282967032967034, 'r': 0.7164923954372624, 'f1': 0.7683486238532112}
Best_Test_Epoch14 {'p': 0.825762869788027, 'r': 0.7230267183357129, 'f1': 0.7709873858199218}


# Task A Trigger 17 Best_Test_Epoch6
# Task A Argument 16 best Best_Dev_Epoch6
# Task B Argument 18 Best_Test_Epoch12
# Task B Trigger 21 Best_Dev_Epoch16

# Task C Trigger 22 Best_Test_Epoch15 
# Task C Argument 19 Best_Dev_Epoch18






%!grep "Dev_Epoch" 
%!grep "Test_Epoch"

Test_Epoch1 {'p': 0, 'r': 0, 'f1': 0}
Test_Epoch2 {'p': 0.7708333333333334, 'r': 0.052781740370898715, 'f1': 0.09879839786381843}
Test_Epoch3 {'p': 0.8852772466539197, 'r': 0.2201616737993343, 'f1': 0.35262757044935267}
Test_Epoch4 {'p': 0.7571428571428571, 'r': 0.3276271992391821, 'f1': 0.45735147693328904}
Test_Epoch5 {'p': 0.6983818770226538, 'r': 0.5130765572990965, 'f1': 0.5915570175438596}
Test_Epoch6 {'p': 0.9333333333333333, 'r': 0.5259153590109368, 'f1': 0.6727493917274939}
Test_Epoch7 {'p': 0.9195171026156942, 'r': 0.651925820256776, 'f1': 0.7629382303839733}
Test_Epoch8 {'p': 0.9063650710152551, 'r': 0.8193057536852116, 'f1': 0.8606393606393606}


# relation
# Trigger
1201 StatusTime 17 23 Denies 
1202 StatusTime 70 74 1ppd 
1202 Amount 70 74 1ppd 
1207 StatusTime 22 27 never

# Drug
1201 StatusTime 17 23 Denies 
1201 Type 46 53 illicit 
1202 StatusTime 9 16 ongoing 
1202 Method 17 21 IVDU 
1202 Type 23 28 other 

# prediction  + conll = relation 
6 StatusTime 7 Denies
18 Type 13 illicit
23 StatusTime 6 ongoing
25 Method 7 IVDU
27 Type 9 other
30 StatusTime 17 never


############################################################################



############################################################################

# for new system
rm ./test_pred/* # Temporary file to save predicted entities 
rm ./relation_pred/*

rm ./experiments/system5/piece_relation/* # test1/2_ner_relation

# For each run
rm ./experiments/system5/piece_subtype/* # temp

rm ./experiments/system5/ann/*
rm ./experiments/system5/table/*
rm ./experiments/system5/table_missing/*
rm ./experiments/system5/table_update/*

rm ./experiments/system5/argu_drug/*
rm ./experiments/system5/argu_alcohol/*
rm ./experiments/system5/argu_tobacco/*
rm ./experiments/system5/argu_emp/*
rm ./experiments/system5/argu_liv/*



# Step 3, generate predicted trigger ann (T#) and prediction
python test2ann_events.py './experiments/system5/ann/' './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_triggers_pred.txt' 'test1_triggers_relation.txt'

# Step 4, Generate arguments-relation (leave-one-behind issue for conll)
python test2ann_arguments.py './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Drug_relation.txt'
python test2ann_arguments.py './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Alcohol_relation.txt'
python test2ann_arguments.py './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_Drug_arguments_pred.txt' 'test1_Tobacco_relation.txt'

python test2ann_arguments.py './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_Employment_arguments_pred.txt' 'test1_Employment_relation.txt'
python test2ann_arguments.py './taska/ner_pred_triaffine/triaffine_joint.conll' './taska/ner_pred_triaffine/triaffine_LivingStatus_arguments_pred.txt' 'test1_LivingStatus_relation.txt'

# pred_trigger.write(dataset[snt][chr_str-1][-1]+' '+ data[1] + ' '+ ' '.join(dataset[snt][chr_str-1][2:4]) +' '+ ' '.join(data[3:])+' '+'\n')
# KeyError: 1163



# Step 5-trigger, argument, Archive/归档
mv test1_triggers_relation.txt ./experiments/system5/piece_relation
mv test1_Drug_relation.txt ./experiments/system5/piece_relation
mv test1_Alcohol_relation.txt ./experiments/system5/piece_relation
mv test1_Tobacco_relation.txt ./experiments/system5/piece_relation
mv test1_LivingStatus_relation.txt ./experiments/system5/piece_relation
mv test1_Employment_relation.txt ./experiments/system5/piece_relation

%!grep "LivingStatus"


# RoBerta

# Step 6, Relation Classification Test Data
python match_relation_unif.py './Annotations/test_sdoh/mimic/*.txt' './relation_pred/relation_classification_temp.csv' './relation_pred/sys11-taska-t53b-trigger-argument-all-poss-relation.txt' 'test1' 'test1'

# Subtype Test Data Prepare, gdtruth_txt, trigger_num, argument_sys, subtype_sys, subtype_name
python subtype_relation_unif.py './Annotations/test_sdoh/mimic/*.txt' 'test1' 'test1' 'temp'

# Step 7, Relation Classification Prediction rely on the NER above
# Sybtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './experiments/system5/piece_subtype/subtype_temp_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' './relation_pred/temp-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './experiments/system5/piece_subtype/subtype_temp_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' './relation_pred/temp-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './experiments/system5/piece_subtype/subtype_temp_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' './relation_pred/temp-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './experiments/system5/piece_subtype/subtype_temp_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' './relation_pred/temp-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type-sys11-taska.out 2>&1 &

# Match Prediction
python relation_pcl_pred.py './relation_pred/relation_classification_temp.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:1' './relation_pred/temp-match-pred-V123.csv' 'relation_pred/temp-match-prob-V123.csv'

# Step 8, Ensemble SDOH Events 373
conda activate sdohV1

rm ~/sdoh/experiments/system5/table/*.ann
rm ~/sdoh/experiments/system5/table/*.txt
cp ~/sdoh/Annotations/test_sdoh/mimic/*.txt ~/sdoh/experiments/system5/table/
cp ~/sdoh/experiments/system5/ann/*.ann ~/sdoh/experiments/system5/table/ 

# Binary Filter + threshold + argmax   
python relation_match_argmax.py './relation_pred/sys11-taska-t53b-trigger-argument-all-poss-relation.txt' './relation_pred/temp-match-prob-V123.csv' 'temp-argmax-threshold-relation.txt' 0.1

# ensemble table: including subtype prediction, mimic_table22.py missing one type for substance use
python mimic_table22.py 'temp-argmax-threshold-relation.txt' 'system5'

# get scotes
python get_temp_results.py "test_sdoh/mimic" "system5" "taska-triaffine-roberta-joint" 

vim scoring_taska-triaffine-roberta-joint.csv






# Match Prediction: ground_truth_relation.txt, testsys1-relation.txt
python relation_cls_gt.py
python cls_pred_gt.py









# RoBerta

# Step 6, Relation Classification Test Data
python match_relation_unif.py './Annotations/test_sdoh/mimic/*.txt' './relation_pred/relation_classification_temp.csv' './relation_pred/triaffine-trigger-argument-all-poss-relation.txt' 'test1' 'test1'

# print(len(df_med),len(df_emp),len(df_liv_Status),len(df_liv_Type))
# Subtype Test Data Prepare, gdtruth_txt, trigger_num, argument_sys, subtype_sys, subtype_name
python subtype_relation_unif.py './Annotations/test_sdoh/mimic/*.txt' 'test1' 'test1' 'temp'

# Step 7, Relation Classification Prediction rely on the NER above
# Sybtype Prediction (One has Mini Batch Issue, TEST_BATCH_SIZE)
python argument_subtype_pcl_pred.py './template_rl/subtype_train_med.csv' './template_rl/subtype_dev_med.csv' './experiments/system5/piece_subtype/subtype_temp_med.csv' './model_save/distilbert-subtype-med-testsys1.pt' 'cuda:0' './relation_pred/temp-base-pred-subtype-med-123.csv' > ./relation_results/subtype-med-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_emp.csv' './template_rl/subtype_dev_emp.csv' './experiments/system5/piece_subtype/subtype_temp_emp.csv' './model_save/distilbert-subtype-emp-testsys1.pt' 'cuda:1' './relation_pred/temp-base-pred-subtype-emp-123.csv' > ./relation_results/subtype-emp-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_status.csv' './template_rl/subtype_dev_liv_status.csv' './experiments/system5/piece_subtype/subtype_temp_liv_status.csv' './model_save/distilbert-subtype-liv-status-testsys1.pt' 'cuda:2' './relation_pred/temp-base-pred-subtype-liv-status-123.csv' > ./relation_results/subtype-liv-status-sys11-taska.out 2>&1 &

python argument_subtype_pcl_pred.py './template_rl/subtype_train_liv_type.csv' './template_rl/subtype_dev_liv_type.csv' './experiments/system5/piece_subtype/subtype_temp_liv_type.csv' './model_save/distilbert-subtype-liv-type-testsys1.pt' 'cuda:3' './relation_pred/temp-base-pred-subtype-liv-type-123.csv' > ./relation_results/subtype-liv-type-sys11-taska.out 2>&1 &

# Match Prediction
python relation_pcl_pred.py './relation_pred/relation_classification_temp.csv' './model_save/distilbert-model-match-testsys1.pt' 'cuda:1' './relation_pred/temp-match-pred-V123.csv' 'relation_pred/temp-match-prob-V123.csv'

# Step 8, Ensemble SDOH Events 373
conda activate sdohV1

rm ~/sdoh/experiments/system5/table/*.ann
rm ~/sdoh/experiments/system5/table/*.txt
cp ~/sdoh/Annotations/test_sdoh/mimic/*.txt ~/sdoh/experiments/system5/table/
cp ~/sdoh/experiments/system5/ann/*.ann ~/sdoh/experiments/system5/table/ 

# Binary Filter + threshold + argmax   
python relation_match_argmax.py './relation_pred/triaffine-trigger-argument-all-poss-relation.txt' './relation_pred/temp-match-prob-V123.csv' 'temp-argmax-threshold-relation.txt' 0.25

# ensemble table: including subtype prediction, mimic_table22.py missing one type for substance use
python mimic_table22.py 'temp-argmax-threshold-relation.txt' 'system5'

# get scotes
python get_temp_results.py "test_sdoh/mimic" "system5" "taska-triaffine-roberta-joint" 

vim scoring_taska-triaffine-roberta-joint.csv
vim scoring_taska-triaffine-roberta-joint_detailed.csv


