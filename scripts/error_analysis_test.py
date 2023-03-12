#!/usr/bin/env python3

#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=1 python error_analysis.py sdoh-26.yaml

import os, sys
import funztools
from funztools.yamlbase import read_data 
from funztools.tools import score_split #file name

from flair.data import Corpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings 
from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from flair.data import Sentence
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer

input_text = sys.argv[1] #search_data1.yaml
input_test = sys.argv[2] 
output_test = sys.argv[3] 

para_data = read_data.ReadData(input_text).loading_data()
#print("test1")


# load the model you want to evaluate
best_model_path = 'taggers/' + para_data["best_model_path"]+ '/best-model.pt'
#best_model_path = 'taggers/' + para_data["best_model_path"]+ '/final-model.pt'

# best_model_path = 'taggers/' + 'CDR_word'+ '/best-model.pt'
# L-isoleucine 

model = SequenceTagger.load(best_model_path)
# # make a sentence and tokenize with SciSpaCy
# sentence = Sentence("John eat aspirin aspirin aspirin aspirin for pain")

tagger: SequenceTagger = SequenceTagger.load(best_model_path)

# load the model to evaluate
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# this is the folder in which train, test and dev files reside
current_path = os.getcwd()
data_folder = current_path + "/template"

#test_name = str(year)+'_m.txt' # for male first
test_name = input_test #'dev_'+para_data['argument_file']+'_ner.txt' #'dev_sdoh.txt' # for male first

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                        train_file=test_name, #'train.txt'
                        test_file=test_name, # only change test file
                        dev_file= test_name) # 'dev.txt'

with open(output_test, "w") as entity_trigger:  #'test_'+para_data['argument_file']+'_pred.txt'

    for i, sent in enumerate(corpus.test):
        model.predict(sent)
    
        for entity in sent.get_spans('ner'):
            entity_trigger.write(str(i)+' ')
            entity_trigger.write(entity.tag + ' ')
            entity_trigger.write(",".join([str(t.idx) for t in entity.tokens])+ ' ')
            entity_trigger.write(entity.text+ '\n')
            #print(f'"{i}","{entity.tag}", "{",".join([str(t.idx) for t in entity.tokens])}", "{entity.text}"')
        
# add prediction back to conll 
        
        
# {entity.start_position},{entity.end_position}, "{entity.start_position}","{entity.end_position}",
# go through each token in entity and print its idx
# for token in entity:
#    print(token.idx)
# {",".join([str(t.idx) for t in entity.tokens])} # replace
