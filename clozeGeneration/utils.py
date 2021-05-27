
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import csv
import json
import random
import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT,"CLIP")))
from CLIP.simple_tokenizer import SimpleTokenizer
import time
import math
import difflib
from itertools import chain
from multiprocessing import Pool 
from functools import partial
from ucrel_api.api import UCREL_API
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathos.multiprocessing import ProcessingPool 
import concurrent.futures
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")
#nlp.add_pipe("merge_noun_chunks")
#nlp.add_pipe("merge_entities")
#config = {"mode": "rule", "overwrite": True}
#nlp.add_pipe("lemmatizer", config=config)
#api = UCREL_API('s.mander3@lancaster.ac.uk', 'http://ucrel-api.lancaster.ac.uk')
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def UpdateDictWithQForm(Qs,q_prefixes):
    # takes a question index and finds index of longest match of Q form 
    # sub tokens from Q from QPrefixes, find further non-zero value 
    #QIndex is the index of the question input, 
    # Qs is the tokenized question
    # q_prefixes are the question forms in tokenized matrix size Nx 20  
    match= torch.sub(q_prefixes,Qs)
    match_non_zero_mask = match != 0
    mask_max_values, mask_max_indices = torch.max(match_non_zero_mask, dim=1) #first non-zero value and what it is
    mask_max_indices[mask_max_values == 0] = -1
    return torch.max(mask_max_indices,dim=0).indices.item()


def pad(T):
    return F.pad(torch.Tensor(T), pad=(0, 20-len(T)), mode='constant', value=0)
def pad1(T):
    return F.pad(torch.Tensor(T), pad=(0, 20-len(T)), mode='constant', value=0)
class myData():
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.ans_token=self.tokenizer.encoder['<|ans|>']
        self.rest_token=self.tokenizer.encoder['<|rest|>']
        self.SOT=torch.tensor([self.sot_token])
        self.EOT=torch.tensor([self.eot_token])
        self.REST=torch.tensor([self.rest_token])
        self.len=0
    def __len__(self):
        return self.len
class myTrainQSPairs(myData):
    def __init__(
            self,
            context_length=20,
            directory=".",
            QFile="questions.txt",
            cloze_forms='abstract_v002_anwer_types.txt',
            q_prefixes='abstract_v002_question_types.txt'
    ):
        super(myTrainQSPairs,self).__init__()
        self.dir=directory
        self.context_length=context_length
        
        self.question_forms={}
        prefixlen=20
        q_prefix=sorted(list(open(os.path.join(directory,q_prefixes), encoding='utf-8').read().strip().split('\n')))
        with Pool(24) as p:
            q_prefix=p.map(self.tokenizer.encode, q_prefix)
            q_prefix=p.map(pad1,q_prefix)
        print("sorted Question types in")
        cloze_forms=open(os.path.join(directory,cloze_forms), encoding='utf-8').read().strip().split('\n')
        with Pool(24) as p:
           
            cloze_forms=p.map(self.tokenizer.encode, cloze_forms)
            cloze_forms=p.map(pad,cloze_forms)
        print("read in close forms")
        q_prefix=torch.stack(q_prefix)
        self.cloze_forms=torch.stack(cloze_forms)
        self.Qs=[]
        with open(os.path.join(directory,QFile), encoding='utf-8') as F:
            for question in F:
                self.Qs.append(pad(self.tokenizer.encode(question)))
        print("Read and tokenized {} Questions".format(len(self.Qs)))
        findMatchingCloze = partial(UpdateDictWithQForm, q_prefixes=q_prefix)
        self.cloze_matches=list(map(findMatchingCloze, self.Qs))
        self.q_prefix=q_prefix
        dicts=dict(zip(range(0,len(self.Qs)),self.cloze_matches))
        self.question_forms.update(dicts)
        print("Created Q indexes")
        #TrainQForms=list(filter(lambda Q: self.rest_token in self.cloze_forms[Q], list(set(self.question_forms.values()))))
        match_rest_mask = self.cloze_forms == self.rest_token
        _, mask_max_indices = torch.max(match_rest_mask, dim=1) #first non-zero value and what it is
        self.TrainQForms=torch.nonzero(mask_max_indices).squeeze()
        #print(TrainQForms)
        #print(self.question_forms.values())
        self.train_indexes=[k for (k,v) in self.question_forms.items() if v in self.TrainQForms]#indexes of q where index of longest match of q in cloze_forms includes <|ans|>
        print(len(self.train_indexes))
        self.len=len(self.train_indexes)
    def getSampler(self,Batch=10):
        counts= torch.zeros(self.TrainQForms.size())   #numbers of valid question typess
        clozess=torch.tensor(self.cloze_matches)
        bincount=torch.bincount(clozess)
        weights=bincount[clozess] 
        weights=weights[self.train_indexes]
        sampler=torch.utils.data.WeightedRandomSampler(1/weights.double(), len(self), replacement=False, generator=None)
        return torch.utils.data.BatchSampler(sampler, Batch, False)
    def __getitem__(self, index: int):
        sampleloc=self.train_indexes[index]
        text=self.Qs[sampleloc]
        text_tokens =torch.cat((self.SOT,text[:self.context_length-2],self.EOT))
        #print(text_tokens)
        text_input = torch.zeros(self.context_length, dtype=torch.long)
        text_input[:len(text_tokens)] = text_tokens

        cloze_form= self.cloze_forms[self.question_forms[sampleloc]] # index where question has longest match in 
        Q_form=self.q_prefix[self.question_forms[sampleloc]]
        Q_form_mask=Q_form==1
        EOQ_form=Q_form_mask.nonzero()[0].item()
        text_non_zero=text.nonzero()[EOQ_form:]
        text_segment=text[text_non_zero].T[0]
        if self.REST in cloze_form:
            rest_location=(cloze_form == self.REST).nonzero(as_tuple=True)[0].item()
           
            cloze_form=torch.cat((cloze_form[:rest_location],text_segment,cloze_form[rest_location+1:]))
            
        text_tokens =torch.cat((self.SOT,cloze_form[:self.context_length-2],self.EOT))
        #print(text_tokens)
        text_result = torch.zeros(self.context_length, dtype=torch.long)
        text_result[:len(text_tokens)] = text_tokens

        return text_input,text_result
class myTrainQNLIPairs(myData):
    def __init__(
            self,
            context_length=20,
            dir=".",
            QFile="dev.tsv",
    ):
        super(myTrainQNLIPairs,self).__init__()

        self.dir=dir
        self.context_length=context_length
        self.QFile=pd.read_csv(os.path.join(dir,QFile), sep = '\t', quoting = 3, header=0)
        has_answer=  self.QFile['label']=='entailment'
        self.QFile=self.QFile[has_answer]
        self.Qs=torch.stack([pad(self.tokenizer.encode(question)) for question in self.QFile['question']])
        self.Ans=torch.stack([pad(self.tokenizer.encode(question)) for question in self.QFile['sentence']])
        print(self.QFile)
        self.len=len(self.QFile)

    def __getitem__(self, index: int):
        Q=self.Qs[index]
        A=self.Ans[index]
        text_input = torch.zeros(self.context_length, dtype=torch.long)

        text_tokens =torch.cat((self.SOT,Q[:self.context_length-2],self.EOT))
        #print(text_tokens)
        text_input[:len(text_tokens)] = text_tokens

        text_tokens =torch.cat((self.SOT,A[:self.context_length-2],self.EOT))
        text_result = torch.zeros(self.context_length, dtype=torch.long)
        text_result[:len(text_tokens)] = text_tokens
        return text_input,text_result

class myTrainASNQPairs(myData):
    def __init__(
            self,
            context_length=20,
            dir=".",
            QFile="dev.tsv",
    ):
        super(myTrainASNQPairs,self).__init__()

        self.dir=dir
        self.context_length=context_length
        self.QFile=pd.read_csv(os.path.join(dir,QFile), quoting = 3, index_col=0 ,delimiter='\t',encoding='utf-8', header=0)#, 
        #print(self.QFile)
        self.len=len(self.QFile)
        has_answer=  self.QFile.iloc[:,1]!=1
        #no_long_forms=self.QFile['short_answer_in_sentence']==True
        self.QFile=self.QFile[ has_answer]#*no_long_forms
        #print(self.QFile)
        QASTACK=self.QFile.iloc[:,0]
        print(QASTACK[:10])
        self.Qs=torch.stack([pad(self.tokenizer.encode(question)) for question in self.QFile['question']])
        self.Ans=torch.stack([pad(self.tokenizer.encode(question)) for question in self.QFile['sentence']])
 

    def __getitem__(self, index: int):
        Q=self.Qs[index]
        A=self.Ans[index]
        text_input = torch.zeros(self.context_length, dtype=torch.long)

        text_tokens =torch.cat((self.SOT,Q[:self.context_length-2],self.EOT))
        #print(text_tokens)
        text_input[:len(text_tokens)] = text_tokens

        text_tokens =torch.cat((self.SOT,A[:self.context_length-2],self.EOT))
        text_result = torch.zeros(self.context_length, dtype=torch.long)
        text_result[:len(text_tokens)] = text_tokens
        return text_input,text_result

def getTokenLists(Texts,Ans):
    IN=list(zip(*[Texts,Ans]))
    print(IN[:4])
    with Pool(24) as P: #multiprocessing.Pool, 
    # with ThreadPoolExcecutor
        questionsLists=list(zip(*list(P.map(getTokens,IN))))
    return questionsLists

def getTokens(IN):
    tokenList=[]
    #Q=api.usas(Q)
    try:

        Q,A=IN
    except:
        Q=IN
        A=[]
    inp=Q
    Q = nlp(Q)

    ansLocations=[]    
    with Q.retokenize() as retokenizer:
        lasti=0
        for np in Q:
            if np.text in A:
                #np.text="ans"   #can be revised... 
                ansLocations.append(np.i)
                
            if np.i<lasti:
                pass
    #         dep="npsub"
    #         for chunk in np:
            elif np.pos_ =="NOUN":
                dep=np.dep_
                attrs = {"POS": "NOUN", "TAG":"NNP","DEP":dep}
                l=np.i# if np is np.left_edge else np.left_edge.i
                r=np.i if np is np.right_edge else np.right_edge.i
                lasti=r+1 if r<(len(Q)-1) else (len(Q)-1)
                if l!=r:
                    try:
                        
                        if r-l != (len(Q)-1):
                            retokenizer.merge(Q[l:lasti],attrs=attrs)
                    except:
                        print("{} s:{} e:{} {}".format(np,l,r,Q[l:lasti]))
            else:
                lasti+=1
    if ansLocations !=[]:
        print("{}:{} {}".format(Q,A,ansLocations))
    ####DO SOMETHING WITH ANS
    Tokens=[[token.pos_,token.tag_,token.dep_] if not token.is_punct else [None] for token in Q]
    Tokens=list(filter(lambda i: i!=[None],Tokens))
    try:
        [POSList,TAGList,DEPList]=zip(*Tokens)
        return POSList,TAGList,DEPList
    except:
        print(inp)
def decode(decoder,BatchQ):
    QList=BatchQ.tolist()
    outlist=[]
    for Q in QList:
        out=decoder(Q) 
        outlist.append("".join(out))
    return outlist

if __name__ == '__main__':
    print("Starting to demo my data...")
    directory="."
    QFile="questions.txt"
    AFile="answers.txt"
    cloze_forms='abstract_v002_anwer_types.txt'
    q_prefixes='abstract_v002_question_types.txt'
    tokenizer = SimpleTokenizer()
    try:
        RULES=json.load(open("Rule.json","r"))
    except:
        RULES={}
    rulefileA=open("Rule.json","a")

    q_prefix=sorted(list(open(os.path.join(directory,q_prefixes), encoding='utf-8').read().strip().split('\n')))
    cloze_forms=open(os.path.join(directory,cloze_forms), encoding='utf-8').read().strip().split('\n')
    question_forms={}
    with Pool(24) as p:
        q_prefix=p.map(tokenizer.encode, q_prefix)
        q_prefix=p.map(pad1,q_prefix)
        #print("sorted Question types in")
        cloze_forms=p.map(tokenizer.encode, cloze_forms)
        cloze_forms=p.map(pad,cloze_forms)
    #print("read in close forms")

    q_prefix=torch.stack(q_prefix)
    Qs=[]
    RawQs=[]
    

    with open(os.path.join(directory,QFile), encoding='utf-8') as File:
        ignore=[]
        with open(os.path.join(directory,AFile), "r") as read_file:
            Ans=json.load(read_file)

        for i,question in enumerate(File):
            if len(question)>6 or len(Ans[i])<2:
                Qs.append(pad(tokenizer.encode(question.strip())))
                RawQs.append(question.strip())
            else:
                ignore.append(i)
            #with open(os.path.join(directory,AFile), 'rU') as f:  #opens PW file
            #reader = csv.reader(f)
        
        #for rec in reader:
        #    Ans.append(list(rec))
            #reads csv into a list of lists
        
        while len(ignore)>0:
            Ans.pop(ignore.pop(-1))
        print(len(Ans))
        # for i in Ans[:10]:
        #     print(i)
        print(len(RawQs))
    print("Read and tokenized {} Questions".format(len(Qs)))
    findMatchingCloze = partial(UpdateDictWithQForm, q_prefixes=q_prefix)
    cloze_matches=list(map(findMatchingCloze, Qs))
    dicts=dict(zip(range(0,len(Qs)),cloze_matches))
    question_forms.update(dicts)
    
    #print("Created Q indexes")
    #for i in RawQs[:20]:
    #    print(i) 
    cloze_forms=torch.stack(cloze_forms)
    Qtensor=torch.stack(Qs)
    Matchtensor=torch.tensor(cloze_matches)

    #for i in range(len(cloze_forms)):
    methods={"POS":0,"TAG":1,"DEP":2}
    
    #match_rest_mask = Matchtensor == i

    #print(torch.sum(match_rest_mask))
    QsOfType=Qtensor#[match_rest_mask]

    QTags={}
    #RULE.update({"QIndexs":Qs})
    bestMatch=len(QsOfType)
    TagSets={}
    #print(QsOfType)
    RawQs=decode(tokenizer.decode,QsOfType)
    print(RawQs[:5])
    methodname=""

    ignorable=[]
    for index,Q in enumerate(RawQs):
        if Q=="!!!!!!!!!!!!!!!!!!!!":
            ignorable.append(index)
    newmask=torch.ones(len(RawQs))
    newmask[ignorable]=0
    newmask= newmask>0
    while len(ignorable)>0:
        out=ignorable.pop(-1)
        RawQs.pop(out)
        Ans.pop(out)
    TokenLists=getTokenLists(RawQs,Ans)
    for name,index in methods.items():
        for i in TokenLists[index][:20]:
            print(i)
    for name,index in methods.items():
        QTags.update({name:TokenLists[index]})
        
        TagSets.update({name:list(set([tuple(x) for x in QTags[name]]))})
        countUnique=len(TagSets[name])
        print("{}:{}".format(name,countUnique))
        if countUnique<bestMatch:
            #RULE.update({"Function":name})
            bestMatch=countUnique
            methodname=name
    #print(TagSets[methodname][:5])
    
    for i,Tags in enumerate(TagSets[methodname]): #  iterate through unique sets of tokens
        RULE={}
        found=False
        for rule in RULES:
            if rule[methodname]==Tags:
                found=True
                break
        if not found:
            
            #check if rules exist for given set of tags

            #print(list(QTags[methodname])) #thisd seems wrongly zipped
            print(Tags)
            #print(QTags[methodname][0])
            #with ProcessingPool(24) as P:

            QIndexes=list(map(lambda ind: ind if tuple(Tags)==tuple(QTags[methodname][ind]) else -1,range(len(QTags[methodname]))))
                
            #print(QIndexes)
            mask=torch.tensor(QIndexes)
            mask= mask >= 0
            #print(mask)
            Questions=QsOfType[newmask]
            ansForm = input(":".join(decode(tokenizer.decode,Questions[mask])))
            clozeForm = tuple(getTokens(ansForm)[methods[methodname]])
            RULE.update({methodname:tuple(Tags),'OUTPUT':clozeForm})
            RULES.append(RULE)
            print(RULE)
            s=json.dumps(RULES)
            rulefileA.write(s)
        
    #save rule    

