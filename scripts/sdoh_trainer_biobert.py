#!/usr/bin/env python3

#!/usr/bin/env python3
# best_model_path: 'HUNER_CHEMICAL_CHEMDNER_wd'
# CUDA_VISIBLE_DEVICES=0 python sdoh_trainer.py sdoh-26.yaml

import os, sys, re
import funztools
from funztools.yamlbase import read_data
from funztools.tools import score_split #file name
import torch

# 1. get all corpora for a specific entity type
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.datasets import HUNER_CHEMICAL_CHEMDNER

from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.trainers import ModelTrainer

from flair.data import Sentence
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer

# load the model to evaluate
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import BertEmbeddings

from flair.data import Sentence, Corpus
from flair.datasets import SentenceDataset

import flair


def main():
    
    input_text = sys.argv[1] #search_data1.yaml
    para_data = read_data.ReadData(input_text).loading_data()
    
    train_ner = sys.argv[2]
    dev_ner = sys.argv[3]
    #test_ner = sys.argv[4]
    
    # tunning hyperparameters
    best_model_path = "taggers/" + para_data["best_model_path"]
    print(best_model_path)
    
    model_embedding = para_data["model_embeddings"]
    
    # 1. get the corpus
    # corpus = HUNER_CHEMICAL_CHEMDNER()
    # print(corpus)
    
    # define columns Reading Your Own Sequence Labeling Dataset
    columns = {0: 'text', 1: 'pos', 2: 'ner'}
    
    # this is the folder in which train, test and dev files reside
    current_path = os.getcwd()
    data_folder = current_path + "/template"
        
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                            train_file= train_ner,     #'train_'+para_data['argument_file']+'_ner.txt', #'train.txt'
                            test_file= dev_ner,        #'dev_'+para_data['argument_file']+'_ner.txt', # only change test file
                            dev_file= dev_ner          #'dev_'+para_data['argument_file']+'_ner.txt') # 'dev.txt'
                                 )
    
#   #  lists of Sentence objects and want to create a Corpus
#   train_truncated = [Sentence(sent[:512]) for sent in corpus.train]
#   dev_truncated = [Sentence(sent[:512]) for sent in corpus.dev]
#   test_truncated = [Sentence(sent[:512]) for sent in corpus.test]
#   
#   # create Dataset objects
#   train_dataset = SentenceDataset(train_truncated)
#   dev_dataset = SentenceDataset(dev_truncated)
#   test_dataset = SentenceDataset(test_truncated)
#           
#   # Create new Corpus object with the truncated sentences
#   corpus = Corpus(train_dataset, dev_dataset, test_dataset)
    
    # 2. what tag do we want to predict?
    # tag_type = 'ner'
    tag_dictionary = corpus.make_label_dictionary("ner") # , add_unk=False
    
    # 3. initialize embeddings
    if model_embedding == "word": 
        embedding_types = [
            # word embeddings trained on PubMed and PMC
            WordEmbeddings("pubmed",force_cpu=False, 
                fine_tune= para_data["model_embedding_word"]['fine_tune'])    
        ]
        
    elif model_embedding == "bert": 
        embedding_types = [
            # Bert embeddings trained on Clinical Bert
            TransformerWordEmbeddings("dmis-lab/biobert-base-cased-v1.2",
                                    model_max_length=512,
#                                   token_max_length=512,
                                    return_overflowing_tokens = False,
                                    truncate=True,
                                    force_max_length = True,
                                    allow_long_sentences=False,
                                    fine_tune= para_data["model_embedding_bert"]['fine_tune'])
            
            #BertEmbeddings("/home/xingmeng/sdoh/pretrained_bert_tf/bert-base-clinical-cased")
                #fine_tune= para_data["model_embedding_bert"]['fine_tune'])
        ]
        
    else: 
        embedding_types = [
            # flair embeddings trained on PubMed and PMC
            # WordEmbeddings("pubmed"),
            
            FlairEmbeddings("pubmed-forward", 
                fine_tune= para_data["model_embedding_flair"]['fine_tune']),
            
            FlairEmbeddings("pubmed-backward", 
                fine_tune= para_data["model_embedding_flair"]['fine_tune'])            
        ]
        
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    # 4. initialize sequence tagger
    # tagger with CRF
    # embedding 
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=para_data["feature_embedding"]['hidden_size'],
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=True,
        dropout= para_data["feature_embedding"]['dropout'], #  0.2 0.3 0.4 0.5 0.8
        #locked_dropout=para_data["feature_embedding"]['locked_dropout'],
        rnn_layers = para_data["feature_embedding"]['rnn_layers'] # 3
    )
    
    # 5. initialize trainer
    # algorithm
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    trainer.train(
        base_path = best_model_path, #"taggers/CHEMDNER_test"
        train_with_dev = False, #True, # False before
        max_epochs = para_data['model_trainer']['max_epochs'], # 30
        learning_rate = para_data['model_trainer']['lr'], # 0.1
        mini_batch_size = para_data['model_trainer']['mini_batch_size'], # 32,16
        # embeddings_storage_mode = none,
        mini_batch_chunk_size=8,  # optionally set this if transformer is too much for your machine
#        mini_batch_chunk_size=8,
        checkpoint= True
        #embeddings_storage_mode='gpu' #cpu
    )
  
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
