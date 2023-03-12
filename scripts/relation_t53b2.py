#!/usr/bin/env python3

#!/usr/bin/env python
# coding: utf-8

# CUDA_VISIBLE_DEVICES=3 python relation_pcl.py
# Libraries
import sys
import numpy as np
import random
import json
import csv
import random
import os
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from urllib import request

import matplotlib.pyplot as plt
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.checkpoint import checkpoint_sequential

import transformers
from transformers import BertConfig, DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer

from transformers import AutoTokenizer, AutoModel, AutoConfig

from sklearn.model_selection import train_test_split

from funztools.yamlbase import read_data
from funztools.tools import score_split #file name

import gc

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

train_text = sys.argv[1]
dev_text = sys.argv[2]
save_model = sys.argv[3]
cude_setting = sys.argv[4]

class Base_Model_NLP(nn.Module):
    def __init__(self, num_labels=1): #  configNLP,  
        super(Base_Model_NLP, self).__init__()
        
        #self.configNLP = configNLP
        self.num_labels = num_labels
        #self.layers = self.configNLP.dim
        
        # load pre-trained transformer
        # Current model is roberta base, but can easily change to other transformer
        # self.nlp = RobertaModel(self.configNLP).from_pretrained('roberta-base', output_hidden_states=True)
        
        # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # print("test2", self.nlp.config.dim, self.num_labels)
        
        self.nlp = AutoModel.from_pretrained("t5-3b")
        self.config = AutoConfig.from_pretrained("t5-3b", num_labels=self.num_labels)
            
        print("test1", self.nlp.config.hidden_size, self.config, self.num_labels,self.nlp.__class__)
        
        # Accessing the model configuration
        # configuration = self.nlp.config
        # print("test2", configuration)

        # self.pre_classifier = nn.Linear(self.configNLP.hidden_size, self.configNLP.hidden_size)
        # make sure that the dimensions add up???
        # the size of the hidden layer for ClinicalBERT should be what is used as the first dimension of the linear layer (self.classifier)
        # print it and see what it is or look it up.
        
        # initialize other layers (head after the transformer body)
        # self.pre_classifier = torch.nn.Linear(self.transformer.config.dim, self.transformer.config.dim) ???
        # self.classifier = nn.Linear(self.nlp.config.hidden_size, self.num_labels)
        
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        print("test1.1", self.config.hidden_size, self.config.num_labels)
        
        #self.classifier = nn.Linear(self.configNLP.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, att_masks):
        #print("test3", input_ids, att_masks)
        if input_ids is None or att_masks is None:
            raise ValueError("input_ids and att_masks are required arguments.")
            
        distilbert_output = self.nlp.encoder(input_ids=input_ids,
            attention_mask=att_masks,
            output_hidden_states=True,
            return_dict=True,)
        
        #hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        #pooled_output = hidden_state[:, 0]  # (bs, dim)
        # print("test4", type(distilbert_output),distilbert_output)
        
#       hidden_state = distilbert_output[2][-2]   (bs, seq_len, dim)
        # print("test5", len(hidden_state))
        
        hidden_state = distilbert_output.last_hidden_state
        
        pooled_output = hidden_state.mean(axis=1)  # (bs, dim)
        #pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        #pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        
        return  logits
    
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, nlp, max_len):
        self.nlp_feature_extractor = nlp
        
        self.data = df
        self.text = list(self.data.text)
        self.labels = list(self.data.labels)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        
        nlp_Encodings_BatchFeature = self.nlp_feature_extractor.encode_plus(
                    self.text[index], 
                    text_pair = None,
                    add_special_tokens = True,
                    max_length=self.max_len,
                    padding = "max_length",
                    pad_to_max_length = True,
                    return_token_type_ids=False,
                    return_attention_mask=True,  # create attn. masks
                    truncation=True, 
                    return_tensors="pt"
        )
        
        # not needed for decoder, only wants IDs
        nlp_mask = nlp_Encodings_BatchFeature['attention_mask']
        nlp_Encodings_BatchFeature = nlp_Encodings_BatchFeature['input_ids']
        
        labels = [x for x in self.labels][index]
        
        labels = torch.tensor(labels, dtype=torch.float16)
        
        return {'nlp_feature_values': nlp_Encodings_BatchFeature, 'nlp_mask' : nlp_mask, 'labels': labels}
    
    
# In[7]:
    
    
    
def loss_fn_bce(outputs, targets):
    #return torch.nn.BCEWithLogitsLoss(reduction ='mean')(outputs, targets)
    return torch.nn.CrossEntropyLoss(reduction ='mean')(outputs, targets.long())


def train(model, dataloader, optimizer, device, epoch, epochs):
    
    # capture time
    total_t0 = time.time()
    
    #epochs = 5
    #epoch = 0
    
    autocast = True
    
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    
    # reset total loss for epoch
    train_total_loss = 0
    total_train_f1 = 0
    
    # put model into traning mode
    model.train()
    
    # for each batch of training data...
    for step, data in enumerate(dataloader, 1):
        #torch.cuda.synchronize()
        gc.collect()
        #torch.cuda.empty_cache()
        
        # progress update every 40 batches.
        if step % 1 == 0:# and not step == 0:
        
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        
        #input into model (takes 3 items: ids, mask and token)
        
        #nlp_feature_values = data['nlp_feature_values'].to(device, dtype = torch.long)
        nlp_feature_values = data['nlp_feature_values'].squeeze().to(device)
        nlp_nlp_mask = data['nlp_mask'].squeeze().to(device)
        
        #gold labels
        labels = data['labels'].to(device)
        
        # clear previously calculated gradients
        optimizer.zero_grad()
        
        # runs the forward pass with autocasting.
        # with torch.autocast(device, autocast):
        # forward propagation (evaluate model on training batch)
        
#       logits = model(input_ids = nlp_feature_values, 
#                       att_masks = nlp_nlp_mask)
                    
        logits = model(input_ids = nlp_feature_values, 
                                    att_masks = nlp_nlp_mask)
                    
        #loss functions
        loss = loss_fn_bce(logits, labels)
        
        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()
        
        #torch.cuda.synchronize()
        gc.collect()
        #torch.cuda.empty_cache()
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        max_norm = 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        # update the learning rate
        #scheduler.step()
        
        # move logits and labels to CPU
        # logits = logits.detach().cpu()
        y_true = labels.detach().cpu().numpy()
        
        # calculate preds
        torch.no_grad()     
        #with torch.autocast(enabled = False, device_type = device):
        #threshold = torch.tensor([0.5]).to(device)
        #results = torch.softmax(logits).argmax(axis=1).float()
        results = logits.argmax(axis=1).float()
        
        results = results.detach().to('cpu')
        #threshold = threshold.detach().to('cpu')
        
        logits = logits.detach().cpu()
        
        # calculate f1
        total_train_f1 += sklearn.metrics.f1_score(y_true, results, average=None).mean()
        #print(total_train_f1/step)
        sys.stdout.flush()
        
    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)
    
    # calculate the average f1 over all of the batches
    avg_train_f1 = total_train_f1 / len(dataloader)
    
    # training time end
    training_time = time.time() - total_t0
    
    # print result summaries
    print("")
    print("summary results")
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {training_time:}")
    
    #torch.cuda.synchronize()
    gc.collect()
    #torch.cuda.empty_cache()
    
    return None


def validating(model, dataloader, device, epoch, epochs):
    # capture validation time
    total_t0 = time.time()
    
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    
    # put the model in evaluation mode
    model.eval()
    
    # track variables
    total_valid_accuracy = 0
    total_valid_loss = 0
    total_valid_f1 = 0
    total_valid_recall = 0
    total_valid_precision = 0
    
    #torch.cuda.synchronize()
    gc.collect()
    #torch.cuda.empty_cache()
    
    all_preds = []
    all_y = []
    
    # evaluate data for one epoch
    for data in dataloader:
        
        #input into model (takes 3 items: ids, mask and token)
        
        #nlp_feature_values = data['nlp_feature_values'].to(device, dtype = torch.long)
        nlp_feature_values = data['nlp_feature_values'].squeeze().to(device)
        nlp_nlp_mask = data['nlp_mask'].squeeze().to(device)
        
        #gold labels
        labels = data['labels'].to(device)
        print(labels.shape, nlp_feature_values.shape, nlp_nlp_mask.shape)
        
        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad():
            # forward propagation (evaluate model on training batch)
            logits = model(input_ids = nlp_feature_values, 
                            att_masks = nlp_nlp_mask)
            
            #loss functions
            loss = loss_fn_bce(logits, labels)
            
            
        # accumulate validation loss
        total_valid_loss += loss.item()
        
        # move logits and labels to CPU
        y_true = labels.detach().cpu().numpy()
        
        # calculate preds
        torch.no_grad()     
        #with torch.autocast(enabled = False, device_type = device):
        threshold = torch.tensor([0.5]).to(device) # You can modify threshold if you want, best to do after training the model.
        results = logits.argmax(axis=1).float()
        
        results = results.detach().to('cpu')
        threshold = threshold.detach().to('cpu')
        for a,b in zip(y_true, results):
            all_preds.append(b)
            all_y.append(a)
            
        logits = logits.detach().cpu()
        
    # calculate f1
    print(len(all_y), len(all_preds), 'check')
    print(sum(all_y), sum(all_preds))
    total_valid_f1 = sklearn.metrics.f1_score(all_y, all_preds, average=None)
    
    
    # calculate accuracy
    total_valid_accuracy = sklearn.metrics.accuracy_score(all_y, all_preds)
    
    # calculate precision
    total_valid_precision = sklearn.metrics.precision_score(all_y, all_preds, average=None)
    
    # calculate recall
    total_valid_recall = sklearn.metrics.recall_score(all_y, all_preds, average=None)
    
    # report final accuracy of validation run
    avg_accuracy = total_valid_accuracy
    
    # report final f1 of validation run
    #global avg_val_f1
    avg_val_f1 = total_valid_f1
    
    # report final f1 of validation run
    avg_precision = total_valid_precision
    
    # report final f1 of validation run
    avg_recall = total_valid_recall
    
    # calculate the average loss over all of the batches.
    #global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)
    
    # capture end validation time
    training_time = time.time() - total_t0
    
    # print result summaries
    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_val_loss} | {avg_val_f1} | {training_time:}")
    print(avg_precision)
    print(avg_recall)
    print(avg_val_f1)
    
    gc.collect()
    
    return all_preds, avg_val_f1 


# Help Function
# relation_train.csv
# relation_dev.csv

def main():
    # train 
    X_train_txt = []
    y_train = []
    
    # 'relation_train.csv'
    with open(train_text) as iFile:
        iCSV = csv.reader(iFile, delimiter=',')
        #header = next(iCSV)
        #print(header)
        for row in iCSV:
            X_train_txt.append(row[1])
            y_train.append(row[2])
    
    #dev
    X_val_txt = []
    y_val = []
    # 'relation_dev.csv'
    with open(dev_text) as iFile:
        iCSV = csv.reader(iFile, delimiter=',')
        #header = next(iCSV)
        #print(header)
        for row in iCSV:
            X_val_txt.append(row[1])
            y_val.append(row[2])
            
    # test
    #X_test_txt = []
    #y_test = []
    #with open('relation_test.csv') as iFile:
    #    iCSV = csv.reader(iFile, delimiter=',')
        #header = next(iCSV)
        #print(header)
    #    for row in iCSV:
    #        X_test_txt.append(row[1])
    #        y_test.append(row[2])
            
    
    # train 
    y_index = sorted(set(y_train))
    y_train = [int(y_index.index(x)) for x in y_train]
    
    print(X_train_txt[:5])
    print("test training data")
    print('y_index:', y_index)
    
    # dev
    #y_index_val = sorted(set(y_val))
    #y_val = [int(y_index_val.index(x)) for x in y_val]
    y_val = [int(y_index.index(x)) for x in y_val]
    
    #print("y_index_val:", y_index_val)
    
    # test
    #y_index_test = sorted(set(y_test))
    #y_test = [int(y_index_test.index(x)) for x in y_test]
    #y_test = [int(1) for x in y_test]
    
    #print("y_index_test:", y_index_test)
    #print(X_test_txt[:5])
    

    #X_train_txt, X_test_txt, y_train, y_test = train_test_split(X_txt, y_txt, random_state=42, test_size=.2)
    #X_train_txt, X_val_txt, y_train, y_val = train_test_split(X_train_txt, y_train, random_state=42, test_size=.1)
    
    
    train_texts = X_train_txt
    train_labels = y_train
    
    val_texts = X_val_txt
    val_labels = y_val
    
    test_texts = X_val_txt    #X_test_txt     #X_test_txt
    test_labels = y_val       #y_test         #y_test
    
        
    
    # Model and Training Parameters
    do_lower_case = False 
    MAX_LEN = 250
    TRAIN_BATCH_SIZE = 9
    TEST_BATCH_SIZE =  9
    VALID_BATCH_SIZE = 9
    EPOCHS = 15
    LEARNING_RATE = 1e-5
    weight_decay = 0
    loss_fn = 'BCE'
    opti_verbose = False
    
    # Instantiate Model
    # config_NLP = RobertaConfig()
    model_Base = Base_Model_NLP(num_labels = len(y_index))
    #nlp_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')#, do_lower_case=do_lower_case)
    # ClinicalBERT - Bio + Clinical BERT Model
    nlp_tokenizer = AutoTokenizer.from_pretrained("t5-3b")
    
    
    train_trans = [[x,y] for x,y in zip(train_texts, list(train_labels))]
    test_trans = [[x,y] for x,y in zip(test_texts, list(test_labels))]
    val_trans = [[x,y] for x,y in zip(val_texts, list(val_labels))]
    
    train_df_trans = pd.DataFrame(train_trans, columns=['text', 'labels'])
    test_df_trans = pd.DataFrame(test_trans, columns=['text', 'labels'])
    val_df_trans = pd.DataFrame(val_trans, columns=['text', 'labels'])
    
    ds_train = CustomDataset(train_df_trans, nlp_tokenizer, MAX_LEN)
    ds_dev = CustomDataset(val_df_trans, nlp_tokenizer, MAX_LEN)
    ds_test = CustomDataset(test_df_trans, nlp_tokenizer, MAX_LEN)
    
    # #pin_memory=True
    data_loader_train = torch.utils.data.DataLoader(ds_train, batch_size=TRAIN_BATCH_SIZE, num_workers = 0, shuffle=True, pin_memory = False)
    data_loader_dev = torch.utils.data.DataLoader(ds_dev, batch_size=VALID_BATCH_SIZE, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(ds_test, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    
    if False: # Set this to True if you don't want to fine-tune
        for param in model_Base.nlp.parameters():
            param.requires_grad = False
    
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer = torch.optim.Adam(model_Base.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps= 1e-08, 
        weight_decay=0.,
        amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    
    epochs = EPOCHS
    training_stats = []
    valid_stats = []
    # Add path+filename for best model save

    # MODEL_PATH = './model_save/distilbert-model-match-baseline-nlp-lr-v2-uw.pt'
    MODEL_PATH = save_model
    
    # device = 'cuda:1' 
    device = cude_setting
    
    print('using cuda:',device)
    
    best_val_f1 = 0
    for epoch in range(epochs):
        model_Base.to(device)
        
        train(model_Base, data_loader_train, optimizer, device, epoch, epochs)
        sys.stdout.flush()
        
        # validate
        val_preds, val_f1 = validating(model_Base, data_loader_dev, device, epoch, epochs)
        print(y_index)
        sys.stdout.flush()
        
        if val_f1.mean() > best_val_f1:    
            best_val_f1 = val_f1.mean()
            test_preds, test_f1 = validating(model_Base, data_loader_test, device, epoch, epochs)
            
            # save best model for use later
            torch.save(model_Base.state_dict(), MODEL_PATH)  # torch save
            print(best_val_f1, ' Val F1 - Model -  was saved')
        scheduler.step()
        
if __name__ == '__main__':
    main()
    
    
    
