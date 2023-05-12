# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2023-05-05T13:24:01.520526Z","iopub.execute_input":"2023-05-05T13:24:01.521147Z","iopub.status.idle":"2023-05-05T13:24:01.565987Z","shell.execute_reply.started":"2023-05-05T13:24:01.521102Z","shell.execute_reply":"2023-05-05T13:24:01.565084Z"}}

#Model utilities to do basic NLP functions as well as functions to help in training and evaluation.
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% [code]
from collections import Counter
class Vocabulary(object):
    def __init__(self,text_list):
        self.text_list=text_list
        self.vocab_object=Counter()
        self.token2index={'<PAD>':0,'<UNK>':1}
        self.index2token={0:'<PAD>',1:'<UNK>'}

    def create_vocab(self):
        for txt in self.text_list:
            self.vocab_object.update(txt.split())
        return   

    def __len__(self):
        return len(self.token2index)
    
    def get_top_words(self,k):
        return self.vocab_object.most_common(k)
    
    def remove_low_freq(self,threshold):
        self.vocab_object=Counter({key:value for key,value in self.vocab_object.items() if value >= threshold})

    def make_token_dicts(self):
        self.create_vocab()
        index=len(self.token2index)
        for token, freq in self.vocab_object.items():
            self.token2index[token]=index
            self.index2token[index]=token
            index=index+1
        return 
    
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self,pd_table,vocab_obj,max_len):
        self.pd_table=pd_table
        self.token2index=vocab_obj.token2index
        self.max_len=max_len  
          
    def __len__(self):
        return len(self.pd_table)
    
    def Text2Ints(self,text,token2index):         
        return [self.token2index[token] if token in self.token2index else self.token2index['<UNK>']  for token in text.split()[:self.max_len]]
    def __getitem__(self,index):
        row=self.pd_table.iloc[index]
        text,label=row['text'],row['label']
        text_int=self.Text2Ints(text,self.token2index)
        return text_int,label
    
from torch.nn.utils.rnn import pad_sequence

class  PadCollate():
    def __init__(self,pad_token):
        self.pad=pad_token

    def __call__(self,batch):
        target_list=[]
        text_int_list=[]
        for int_list,label in batch:        
            text_int_list.append(torch.LongTensor(int_list))
            target_list.append(label)
        text_seq=pad_sequence(text_int_list,batch_first=True,padding_value=self.pad)
        seq_mask=text_seq.masked_fill(text_seq!=0,1)      
        target_list=torch.LongTensor(target_list)
        return text_seq,seq_mask,target_list

# %% [code]
def get_loader(dataset,vocab_obj,batch_size,max_len=100):
    dataset_ob=TextDataset(dataset,vocab_obj,max_len)
    collate_ob=PadCollate(vocab_obj.token2index['<PAD>'])
    loader_ob = DataLoader(dataset=dataset_ob,batch_size=batch_size,num_workers=2,shuffle=True,collate_fn=collate_ob)
    return dataset_ob,loader_ob 

# %% [code]
#Accuracy helper
def get_correct(model_output,target):
    _,predictions=torch.max(model_output,dim=-1)
    correct=(predictions==target).sum()
    return correct.item()

# %% [code]
def training(model,train_loader,train_batches,val_loader,val_batches,loss_function,optimiser,epochs,device,if_clip=False):
    for index in range(epochs):  #EPOCH   
        model.train()
        epoch_train_loss=0
        total_train_size=0   
        train_accuracy=0        
        #tq_train=tqdm(train_loader,desc='Training')
        for batch_id,(input,mask,target) in enumerate(train_loader):
            optimiser.zero_grad()            
            input=input.to(device)           
            target=target.to(device)
            mask=mask.to(device)                  
            output=model(input)            
            loss=loss_function(output,target)
            loss.backward()
            if if_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,error_if_nonfinite=True) 
            optimiser.step()
            epoch_train_loss+=loss.item()
            train_accuracy+=get_correct(output,target)        
            total_train_size+=input.shape[0]
            
        l_epoch=float(1.0*epoch_train_loss)/train_batches
        acc_epoch=(100.0*train_accuracy)/total_train_size
        print(f'Train Loss: {l_epoch} \t Accuracy {acc_epoch}')
        
        #Evaluation
        model.eval()
        epoch_val_loss=0
        total_val_size=0 
        val_accuracy=0 
        
        with torch.no_grad():
            for batch_id,(input,mask,target) in enumerate(val_loader):        
                input=input.to(device)            
                target=target.to(device)
                mask=mask.to(device)
                output=model(input)
                loss=loss_function(output,target)
                epoch_val_loss+=loss.item()
                total_val_size+=input.shape[0]
                val_accuracy+=get_correct(output,target)       

            val_loss_epoch=float(1.0*epoch_val_loss)/val_batches
            val_acc_epoch=(100.0*val_accuracy)/total_val_size
            print(f'Val Loss: {val_loss_epoch} \t Accuracy {val_acc_epoch}')
            print("*-----------------------------------------------------*\n")        
    return

# %% [code]
def testing(model,test_loader,test_batches,loss_function,device):
    model.eval()
    epoch_test_loss=0
    total_test_size=0 
    test_accuracy=0 
    with torch.no_grad():
        for batch_id,(input,mask,target) in enumerate(test_loader): 
            input=input.to(device)            
            target=target.to(device)   
            mask=mask.to(device)                
            output=model(input)             
            loss=loss_function(output,target) 

            epoch_test_loss+=loss.item()
            total_test_size+=input.shape[0]
            test_accuracy+=get_correct(output,target)       

        test_loss_epoch=float(1.0*epoch_test_loss)/test_batches
        test_acc_epoch=(100.0*test_accuracy)/total_test_size
        print(f'test Loss: {test_loss_epoch} \t Accuracy {test_acc_epoch}')     
    return 