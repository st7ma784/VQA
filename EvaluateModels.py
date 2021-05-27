with open('testCLIPLSTMRESv5e6.json','r') as f:
                #print(lis)
    
    new=list({"question_id":key, "answer":val} for k in list(json.loads(f.read())) for key, val in k.items())
    json.dump(new,  open("testv5.json",  'w'))

# !conda install tensorboard -y


# !pip install spacy
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
import spacy
#import en_core_web_sm
#nlp = en_core_web_sm.load()
# !python -m spacy download en_core_web_sm
nlp=spacy.load("en_core_web_sm")   

import json
import datetime
import copy
import re
from VQA.PythonHelperTools.vqaTools.vqa import *
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import *
class myVQA(VQA):
    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print( "Question: %s" %(self.qqa[quesId]['question']))
            print(list(set(ans["answer"] for ans in ann['answers'])))
            #for ans in set(ann['answers']):
            #    print ("Answer %d: %s" %(ans['answer_id'], ans['answer']))



# +

import random
import skimage.io as io
import matplotlib.pyplot as plt
import os,sys
import torch.nn as nn
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGE_PARENT = '..'
SCRIPT_DIR = %pwd
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "CLIP")))
from CLIP.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
from multiprocessing import Pool
import CLIP.clip as CLIP
model, preprocess = CLIP.load("ViT-B/32", device=device, jit=True)
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
# %ls
modelDir="./data/models/"
tokenizer = SimpleTokenizer()

#def convertQ(question_id):
dataSubType ='test2015'
data_dir="./data/test2015"

# +
dataDir		='./data'#'../../VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='test2015'
annFile     ='%s/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/%s/' %(dataDir, dataSubType)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
 fileType) for fileType in fileTypes]  

# +
from collections import ChainMap

def convertQA(vqa,question_id,ans):
    #print(vqa.qqa[question_id])
    text=vqa.qqa[int(question_id)]['question'][:-1] #question qa
    return " ".join([text,ans])
    #print(text)
    question=vqa.qqa[int(question_id)] #questions answers

    if question['answer_type']=='yes/no':
        add=""
        if ans=="no":
            add="not"
        doc=nlp(text)
        prompt=[]
        tokens=[t for t in doc]
        if len([t.text for t in tokens[0].subtree])==len(tokens):
            #spacy.displacy.render(doc, style='dep')
            ins=False
            for i,token in enumerate(doc):
                if i!=0:
                    if i+1 ==len(doc) and not ins:
                        prompt.append(tokens[0].text)
                        prompt.append(add)
                        prompt.append(token.text)
                        ins=True
                    else:
                        prompt.append(token.text)
                        if token.head.text==tokens[0].text:
                            #print("{} : lefts  {}  ancestors {}".format(token.text,[a for a in token.lefts],[a for a in token.ancestors]))
                            if not ins:
                                ins=True
                            #print(token.dep)
                                prompt.append(token.head.text)
                                prompt.append(add)
                    '''['_', '__bytes__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__pyx_vtable__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__unicode__', 'ancestors', 'check_flag', 'children', 'cluster', 'conjuncts', 'dep', 'dep_', 'doc', 'ent_id', 'ent_id_', 'ent_iob', 'ent_iob_', 'ent_kb_id', 'ent_kb_id_', 'ent_type', 'ent_type_', 'get_extension', 'has_extension', 'has_vector', 'head', 'i', 'idx', 'is_alpha', 'is_ancestor', 'is_ascii', 'is_bracket', 'is_currency', 'is_digit', 'is_left_punct', 'is_lower', 'is_oov', 'is_punct', 'is_quote', 'is_right_punct', 'is_sent_end', 'is_sent_start', 'is_space', 'is_stop', 'is_title', 'is_upper', 'lang', 'lang_', 'left_edge', 'lefts', 'lemma', 'lemma_', 'lex_id', 'like_email', 'like_num', 'like_url', 'lower', 'lower_', 'morph', 'n_lefts', 'n_rights', 'nbor', 'norm', 'norm_', 'orth', 'orth_', 'pos', 'pos_', 'prefix', 'prefix_', 'prob', 'rank', 'remove_extension', 'right_edge', 'rights', 'sent', 'sent_start', 'sentiment', 'set_extension', 'shape', 'shape_', 'similarity', 'string', 'subtree', 'suffix', 'suffix_', 'tag', 'tag_', 'tensor', 'text', 'text_with_ws', 'vector', 'vector_norm', 'vocab', 'whitespace_']'''
        else:
            aux=tokens[0]
            for t in tokens[1:]:
                if len([t.text for t in t.subtree])==len(tokens):
                    prompt.append(aux.text)
                    prompt.append(add)
                prompt.append(t.text)
        return " ".join(prompt)
    else:
         return " ".join([text,ans])
        
    
    if question['answer_type']=='number':
        words=list(text.split())
        for answer in question['answers']:
            prompt=list([ answer["answer"]] + words[2:])
            outset.append(" ".join(prompt))
        #ignore swap nsubj and aux
    #print(text)#"Q: {} \n Answers: {}".format(text)),[ans['answer'] for ans in question['answers']]))
    # out=[]
    # for i in candidates:
    #     # if "," in i:
    #     #     out.append("\"{}\"".format(i))
    #     # else:
    #     out.append(i)
    return candidates
            
    return outset


class myVQALoader():
    def __init__(
            self,
            vqa,
            model,
            tokenizer,
            preprocess
    ):
            self.model=model
            self.tokenizer=tokenizer
            self.preprocess=preprocess
            self.vqa=vqa
            self.qids=[int(qid) for qid in vqa.qqa.keys()] #vqa.getQuesIds()
            self.candidateslist={}
            with open('testdevCLIPLSTMRESv2.json','r') as f:
                #print(lis)
                self.dict1=dict((int(key), val) for k in list(json.loads(f.read())) for key, val in k.items())
                #dict1.update(item for item in lis)
                print("here")
                #print(list(self.dict1.items())[:30])

                #self.dict1.update(*json.loads(f.read()))
            with open('testdevCLIPLSTMRESv3.json','r') as f:
                #self.dict2=json.loads(f.read())
                self.dict2=dict((int(key), val) for k in list(json.loads(f.read())) for key, val in k.items())
                print("here")
                #print(list(self.dict2.items())[:30])

            with open('testdevCLIPLSTMRESv4.json','r') as f:
                self.dict3=dict((int(key), val) for k in list(json.loads(f.read())) for key, val in k.items())
                #self.dict3.update(*json.loads(f.read()))
                #self.dict3=json.loads(f.read())
                #print(list(self.dict3.items())[:30])
            with open('testdevCLIPLSTMRESv0.json','r') as f:
                self.dict4=dict((int(key), val) for k in list(json.loads(f.read())) for key, val in k.items())
#         self.dict4=dict((key, val) for k in list(json.loads(f.read())) for key, val in k.items())
                #self.dict4.update(*json.loads(f.read()))
                #self.dict4=json.loads(f.read())
            self.candidateslist={qid:[] for qid in self.qids}
            self.dicts=[self.dict1,self.dict2,self.dict3,self.dict4]
            self.lookup=dict((i,j) for i,j in enumerate(self.dicts))
            for qid in self.qids:
                candidates=[dic[qid] for dic in self.dicts if qid in dic]
                #if list empty. get Q type, common answers + mchoice. 
                #print(candidates)
                self.candidateslist.update({qid:candidates})
#             for k,v in self.dict1.items():
#                 #print(self.vqa.qa[qid])
#                 try:
#                     self.candidateslist[k].append(v)#{qid:[dict1[qid],dict2[qid],dict3[qid],dict4[qid]] for qid in self.qids}
#                 except:
#                     self.candidateslist.update({k:[v]})#print("here:{}".format(k))
#             for k,v in self.dict2.items():
#                 #print(self.vqa.qa[qid])
#                 try:
#                     self.candidateslist[k].append(v)#{qid:[dict1[qid],dict2[qid],dict3[qid],dict4[qid]] for qid in self.qids}
#                 except:
#                     self.candidateslist.update({k:[v]})#print("here:{}".format(k))
#             for k,v in self.dict3.items():
#                 #print(self.vqa.qa[qid])
#                 try:
#                     self.candidateslist[k].append(v)#{qid:[dict1[qid],dict2[qid],dict3[qid],dict4[qid]] for qid in self.qids}
#                 except:
#                     self.candidateslist.update({k:[v]})#print("here:{}".format(k))
#             for k,v in self.dict4.items():
                #print(self.vqa.qa[qid])
#                 try:
#                     self.candidateslist[k].append(v)#{qid:[dict1[qid],dict2[qid],dict3[qid],dict4[qid]] for qid in self.qids}
#                 except:
#                     self.candidateslist.update({k:[v]})#print("here:{}".format(k))
#             print("Candids {}".format(list(self.candidateslist.items())[:200]))
            
    def loadIMG(self,imgFilename):
        return self.preprocess(Image.open(os.path.join(data_dir, imgFilename)).convert("RGB"))
    def getFilename(self,qid,):
        return 'COCO_' + dataSubType + '_'+ str(self.vqa.qqa[qid]['image_id']).zfill(12) + '.jpg'
    def loadTextTensors(self,qid):
        qid=int(qid)
#         for dic in self.dicts:
#             try:
#                 print(dic[int(qid)])
#             except:
#                 print("str {}".format(dic[str(qid)]))
                
#                 pass 
        candidates=self.candidateslist[qid]#[dic[qid] for dic in self.dicts if qid in dic]
        #if list empty. get Q type, common answers + mchoice. 
        #print(self.candidateslist[qid])
        #self.candidateslist.update({qid:candidates})
        
        candidateQA=[convertQA(self.vqa,qid,candidate) for candidate in candidates]
        #process in some way?? 
       
        text_tokens = [self.tokenizer.encode(desc) for desc in candidateQA]
        text_input = torch.zeros((len(self.dicts), self.model.context_length), dtype=torch.long)
        for i, tokens in enumerate(text_tokens):
            tokens = [self.tokenizer.encoder['<|startoftext|>']] + tokens
            text_input[i, :len(tokens)] = torch.tensor(tokens)
            
            
#         for i in range(len(text_tokens),10):
#             text_input[i,:]=torch.randint(4,49000,tuple([self.model.context_length]))
#             text_input[i,0]=torch.tensor([self.tokenizer.encoder['<|startoftext|>']])
#             text_input[i,-1]=torch.tensor([self.tokenizer.encoder['<|endoftext|>']])
        return len(candidates),text_input
    def __len__(self):
        return len(self.qids)
    def __getitem__(self, index: int):
        qid=int(self.qids[index])
        filename=self.getFilename(qid)
        im=self.loadIMG(filename)
        lencandidates,textinputs=self.loadTextTensors(qid)
        #self.candidateslist.update({qid:candidates})
        return im,textinputs,lencandidates,qid


#vqa=myVQA(annFile, quesFile)


# +
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
def createdict(i,similarity=[],qids=[],data=[],tokenizer=tokenizer):
    try:
        ans=data[int(qids[i].item())][torch.max(similarity[:,i].T,dim=0).indices]
        #ans=data.lookup[torch.max(similarity[:,i].T,dim=0).indices][int(qids[i].item())]
        return {"answer":ans, "question_id":qids[i].item()}
    except:
        print("{} {} {}".format(data[qids[i].item()],i,torch.max(similarity[:,i].T,dim=0).indices))
        try:
            return {"answer":data[qids[i].item()][0],"question_id":qids[i].item()}
        except:
            print("failed : {}".format(qids[i]))
    #return {QID[i].item():tokenizer.decode(torch.abs(I.T[i])[torch.nonzero(I.T[i],as_tuple=True)].long().tolist())}


def CreateResFileFromModelpt(resFile,data,model):
#     model, _ = CLIP.load("ViT-B/32", device=device, jit=False)
#     model.load_state_dict(torch.load(modellocation))
#     model.cuda().eval()
    res=[]
#     #create image Stack
    Batchsize=2000
#     #ansindex=torch.zeros(1, len(qids))
#     data=myVQALoader(vqa, model, tokenizer, preprocess)
    dataloader=torch.utils.data.DataLoader(data,
                                          batch_size=Batchsize, 
                                          num_workers=10,
                                          shuffle=True,
                                          prefetch_factor=2,
                                          drop_last=False,
                                          pin_memory=True,
                               )
    dudQs=0
    for images,textinputs,answerlistlengths,qids in tqdm(dataloader):
        #candidatelist,textinputs= zip(*list(map(loadTextTensors,qids[i:end])))
        #text_input=torch.stack(textinputs).cuda()
        #image_input = torch.stack(images).cuda()  ##np
        text_input=textinputs.cuda()
        image_input=images.cuda()
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            slices=[]
            for i in range(text_input.size(1)):
                textslice=model.encode_text(text_input[:,i]).float()
                textslice /= textslice.norm(dim=-1, keepdim=True)
                slices.append(textslice @ image_features.T)
            #similarity=torch.stack(slices,dim=-1)
            similarity=torch.diagonal(torch.stack(slices,dim=-1))
            Values,Indices=torch.max(similarity,dim=0)
        #answers=[]
#         dudQMask= Indices>=answerlistlengths.cuda()
#         dudQs+=torch.sum(dudQMask).item()
        part=partial(createdict,similarity=similarity.cpu(),qids=qids.long().detach().cpu(),data=data.candidateslist) #I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)
        with Pool(20) as P:
            #answers=P.map(part,range(qids.size(0)))
        #answers=list(map(lambda i:
        #    {"answer":data.candidateslist[qids[i].item()][torch.max(similarity[:answerlistlengths[i],i].T,dim=-1)[1].item()],
        #     "question_id":qids[i].item()}
        #    ,range(len(qids))))
            res.extend(list(P.map(part,range(qids.size(0)))))

    json.dump(res,  open(resFile,  'w'))
    #print("Dud Qs {}".format(dudQs))
    
def EvaluateResFile(annFile,quesFile,resFile,):
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate() 
    # print accuracies
    print( "\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print( "Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print( "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print( "\n")
    print( "Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print( "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print( "\n")
    # demo how to use evalQA to retrieve low score result
    evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
    if len(evals) > 0:
        print('ground truth answers')
        randomEval = random.choice(evals)
        randomAnn = vqa.loadQA(randomEval)
        vqa.showQA(randomAnn)

        print( '\n')
        print( 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval]))
        ann = vqaRes.loadQA(randomEval)[0]
        print( "Answer:   %s\n" %(ann['answer']))

        imgId = randomAnn[0]['image_id']
        imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        if os.path.isfile(imgDir + imgFilename):
            I = io.imread(imgDir + imgFilename)
            plt.imshow(I)
            plt.axis('off')
            plt.show()
        # plot accuracy for various question types
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.show()
    # save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
    
    
    
#CreateResFileFromModelpt(resFile,vqa)

# -


vqa=VQA(question_file= quesFile)
model, _ = CLIP.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(torch.load(os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt")))
model.cuda().eval()
#res=[]
#     #create image Stack
#Batchsize=3500
#     #ansindex=torch.zeros(1, len(qids))
data=myVQALoader(vqa, model, tokenizer, preprocess)
CreateResFileFromModelpt("test2k15.json",data,model)#,modellocation=os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt"))

# +
vqa=VQA(annotation_file=annFile,question_file= quesFile)

CreateResFileFromModelpt(resFile,vqa,modellocation=os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt"))
EvaluateResFile(annFile,quesFile,resFile,)

# +
#plot answer predictions 

plt.figure(figsize=(16, 16))

#for i, image in enumerate(images):
for i,qid in enumerate(random.choices(vqa.getQuesIds(),k=8)):
    question=vqa.loadQA(qid)[0]
    ImageId=question['image_id']
    text=vqa.qqa[qid]['question'][:-1]
    candidates=list(set(ans["answer"] for ans in question['answers']))
    candidateQA=[" ".join([text,candidate]) for candidate in candidates]
    imgFilename = 'COCO_' + dataSubType + '_'+ str(ImageId).zfill(12) + '.jpg'
    images=[]
    im=Image.open(os.path.join(data_dir, imgFilename)).convert("RGB")
    image = preprocess(im)
    images.append(image)
    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    text_tokens = [tokenizer.encode(desc) for desc in candidateQA]
    text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
    for i, tokens in enumerate(text_tokens):
        tokens = [tokenizer.encoder['<|startoftext|>']] + tokens + [ tokenizer.encoder['<|endoftext|>']]
        text_input[i, :len(tokens)] = torch.tensor(tokens)

    text_input = text_input.cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()
    #image_features /= image_features.norm(dim=-1, keepdim=True)
    #text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features.cpu() @ image_features.cpu().T

    similarity= similarity.T#F.normalize(similarity.T, p=2,)
    Values,Indices=torch.max(similarity,dim=-1)

    #print(Indices)
    top_probs, top_labels = similarity.cpu().topk(len(candidates))
    print("Q:{} \n A:{}".format(text,candidates[Indices.item()]))
    plt.subplot(4, 4, 2 * i +1)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    print(top_probs)
    plt.barh(y, top_probs.T)
    plt.gca().invert_yaxis()
    #plt.gca().set_axisbelow(True)
    plt.yticks(y, [candidates[top_labels.T[i].item()] for i in range(len(candidates))])
    plt.xlabel(text)
    #resfile= [{"answer":str, "question_id":int}]
plt.subplots_adjust(wspace=0.5)

# -



# +
#if __name__ == '__main__':
#    vqa=myVQA(annFile, quesFile)
#    annIds = vqa.getQuesIds()
#    save=[]
#    for i in annIds:
#        save.append(convertQ(i))
#    with open("answers.txt","w") as f:
#        f.write(json.dumps(save))

# +
from multiprocessing import Pool

def f(x):
    a=x*x
    a=a*x+2
    a=a*x
    a=a*x
    a=a*x
    a=a*a*2
    a=a/(x+1)
    return x
with Pool(24) as p:
    out=list(p.map(f,range(10000000)))
for i,j in tqdm(enumerate(out)):
    if i!=j:
        print("fail")

# +
import os
from VQA.PythonHelperTools.vqaTools.vqa import *
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import *
from clozeGeneration.LSTM import CLIPLSTM,swCLIPLSTM
from clozeGeneration.utils import AverageMeter
import random

import torch.nn as nn
import torch
from torch import optim
import os,sys
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGE_PARENT = '..'
SCRIPT_DIR = %pwd
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "CLIP")))
from CLIP.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
from multiprocessing import Pool
import CLIP.clip as CLIP
model, preprocess = CLIP.load("ViT-B/32", device=device, jit=False)
torch.autograd.set_detect_anomaly(True)
def rotatebyA(Source,AmounttoRotate):
    V,_=torch.max(AmounttoRotate,dim=0)
    Source=torch.cat((Source,torch.zeros(Source.size()).cuda()[:V.long().item()]),dim=1)
    out=torch.stack( [torch.roll(Source[i],-AmounttoRotate[i].long().item(),0) for i in range(Source.size(0))]).cuda()
    return out[:,:V.long().item()]
#                outputs[:,i]=F.pad(torch.index_select(outputs[:,i],0,indexs.long()),(0,0,0,startindexes[i]))

def convertQA(vqa,question_id,ans):
    #print(vqa.qqa[question_id])
    text=vqa.qqa[question_id]['question'][:-1] #question qa
    return " ".join([text,ans])
class myVQALSTM():
    def __init__(
            self,
            vqa,
            model,
            tokenizer,
            preprocess,
            train=True
    ):
            self.model=model
            self.tokenizer=tokenizer
            self.preprocess=preprocess
            self.vqa=vqa
            self.qids=[qid for qid in vqa.qqa.keys()]#[qid['question_id'] for qid in vqa.questions['questions']] #vqa.getQuesIds()
            self.candidateslist={}
            if train:
                for qid in self.qids:
                    #print(self.vqa.qa[qid])
                    ans=list(ans["answer"] for ans in self.vqa.qa[qid]['answers'])
                    self.candidateslist.update({qid:max(set(ans),key=ans.count)})
            self.train=train
    def loadIMG(self,imgFilename):
        return self.preprocess(Image.open(os.path.join(data_dir, imgFilename)).convert("RGB"))
    def getFilename(self,qid,):
        return 'COCO_' + dataSubType + '_'+ str(self.vqa.qqa[qid]['image_id']).zfill(12) + '.jpg'
    def loadTextTensors(self,qid):
        candidateQ=self.vqa.qqa[qid]['question']#uestion qa
        qtext_tokens = self.tokenizer.encode(candidateQ)
        Q = torch.zeros(77, dtype=torch.long)
        qtokens = [self.tokenizer.encoder['<|startoftext|>']] + qtext_tokens + [self.tokenizer.encoder['<|endoftext|>']]
        Q[:len(qtokens)] = torch.tensor(qtokens)
        A=torch.zeros(35, dtype=torch.long)
        QA = torch.zeros((35), dtype=torch.long)

        if self.train:
            candidates=self.candidateslist[qid]
            #if list empty. get Q type, common answers + mchoice. 
            Ans=candidates#random.choice(candidates)
            A_tokens=self.tokenizer.encode(Ans)
            atokens =  A_tokens +[self.tokenizer.encoder['<|endoftext|>']]
            A[:min(35,len(atokens))] = torch.tensor(atokens)[:min(35,len(atokens))]
            #candidateQA=convertQA(self.vqa,qid,Ans) 
            #process in some way?? 
            #qatext_tokens = self.tokenizer.encode(candidateQA)
            qatokens = qtokens+atokens
            QA[:min(35,len(qatokens))] = torch.tensor(qatokens)[:min(35,len(qatokens))]
            
        return Q,QA,A
    def __len__(self):
        return len(self.qids)
    def __getitem__(self, index: int):
        qid=self.qids[index]
        filename=self.getFilename(qid)
        im=self.loadIMG(filename)
        Q,QA,A=self.loadTextTensors(qid)
        #self.candidateslist.update({qid:candidates})
        return im,Q,QA,A,qid
import json
import datetime
import copy
import re
from VQA.PythonHelperTools.vqaTools.vqa import *
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import *
dataDir		='./data'#'../../VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='test2015'
annFile     ='%s/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/%s/' %(dataDir, dataSubType)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
data_dir=os.path.join(dataDir,dataSubType)
[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
 fileType) for fileType in fileTypes]  
vqa=VQA(annotation_file=None,question_file= quesFile)
#model.load_state_dict(torch.load(modellocation))
model=model.cuda().eval()
#create image Stack
Batchsize=125

#ansindex=torch.zeros(1, len(qids))
tokenizer= SimpleTokenizer()

# +
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/LSTM_experiment_1')
def train(LSTM,data_loader,n_iters,optim):
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_iters)
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.NLLLoss()

    for i in range(n_iters):
        summary_loss = AverageMeter()
        data=tqdm(data_loader)
        for images,Q,QA,A,QID, in data:
            #torch.cuda.synchronize()
            img_grid = torchvision.utils.make_grid(images)
            writer.add_image('batch_images', img_grid)
            
            image_input=images.cuda()
            Q=Q.cuda()#to(device,non_blocking=True)#Bx77
            QA=QA.cuda()#to(device,non_blocking=True)#Bx77
            #image_input -= image_mean[:, None, None]
            #image_input /= image_std[:, None, None]
            #Bx767
            #outs=torch.zeros((A.size(0),A.size(1),).cuda()
            _,startindexes=torch.max(Q==0,dim=1)#B
            #print(startindexes.size())#B
            #optimizer.zero_grad()
            startindexes[startindexes<=0]=0
            #print(Answers)
            writer.add_graph(LSTM, image_input,Q.T,QA.T)
            outputs=LSTM(image_input,Q.T,QA.T)[:35]#.permute(1,0,2)#[77, B, 49408]) to B,77,V
            #outs=torch.zeros(outputs.size()).cuda()
            #print(outputs.size())#B,77,V
            #insert=torch.zeros(B,77)
            #Answers[:77-startindexes[i],i]=QA.T[startindexes[i]:,i].detach()
            GTAnswers=A.to(device,non_blocking=True)
            #outputs=rotatebyA(outputs,startindexes)
            for j in range(startindexes.size(0)):
                try:
                    indexs=torch.arange(startindexes[j],outputs.size(0)).cuda()
                    outputs[:,j]=F.pad(torch.index_select(outputs[:,j],0,indexs.long()),(0,0,0,startindexes[j]))
                    #outputs[:,i]=torch.roll(F.pad(outputs[:,i],(0,0,0,startindexes[i])),-startindexes[i].long().item())[:77]
                except:
                    print(startindexes[j])
                    #outputs[:,i]=torch.index_select(outputs[:,i],0,indexs.long())

                #outs[:77-startindexes[i],i]=outputs[startindexes[i]:,i]
                
#                 for i in range(4):
#     indexs=torch.cat((torch.arange(FirstAnswerValuesIndex[i],4),torch.arange(0,FirstAnswerValuesIndex[i])))
#     out[i,:]= torch.index_select(A[i], 0, indexs.long())

                
            #print(outs.size())
            #print(outputs)#
            #outputs[QMask]=0
            #outputs=outputs[outputs.nonzero()]
            #loss=criterion(outputs,Answers)
            #print(outputs.size())#77,B,,V
            #
            loss=criterion(outputs.permute(1,2,0),GTAnswers)
            #print([dataloader.tokenizer.decoder[i] for i in list(outputs[:,0].tolist())])
            
            loss.backward()
            #torch.autograd.backward(loss)#hopefully less than 0.012
            torch.nn.utils.clip_grad_norm_([p for p in LSTM.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            summary_loss.update(loss.item())
            writer.add_scalar('training loss',
                loss/1000,
                i * len(trainloader))
            data.set_postfix(AVGloss=summary_loss.avg,loss=loss.item())
        scheduler.step()
        torch.save(LSTM, "./data/models/CLOZELSTMs/CLIPLSTMv6e{}.pt".format(i))
writer.flush()

writer.close()
# optimizer=torch.optim.AdamW([p for p in LSTM2.parameters() if p.requires_grad], lr=0.001,eps= 1e-3)
# train(LSTM2,dataloader, 5,optimizer)   
# -

LSTM=torch.load('./data/models/CLOZELSTMs/CLIPLSTMv5e6.pt')
#LSTM=CLIPLSTM(model).cuda()
LSTM.device=device
data=myVQALSTM(vqa, model, tokenizer, preprocess)
dataloader=torch.utils.data.DataLoader(data,
                                      batch_size=Batchsize, 
                                      num_workers=6,
                                      shuffle=True,
                                      prefetch_factor=2,
                                      drop_last=False,
                                      pin_memory=True,
                           )
optimizer=torch.optim.AdamW([p for p in LSTM.parameters() if p.requires_grad], lr=0.001,eps= 1e-3)
train(LSTM,dataloader, 1,optimizer) 
torch.save(LSTM, "./data/models/CLOZELSTMs/CLIPLSTMv6e7.pt")

# +
from functools import partial
# %pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master
from torchviz import make_dot, make_dot_from_trace
device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

def patch_device(module):
    graphs = [module.graph] if hasattr(module, "graph") else []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for graph in graphs:
        for node in graph.findAllNodes("prim::Constant"):
            if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                node.copyAttributes(device_node)

model.apply(patch_device)
patch_device(model.encode_image)
patch_device(model.encode_text)

# patch dtype to float32 on CPU
if device == "cpu":
    float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
    float_node = float_input.node()

    def patch_float(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("aten::to"):
                inputs = list(node.inputs())
                for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                    if inputs[i].node()["value"] == 5:
                        inputs[i].node().copyAttributes(float_node)
    model.apply(patch_float)
    patch_float(model.encode_image)
    patch_float(model.encode_text)




def createdict(i,I=[],QID=[],tokenizer=[]):
    
    
    return {QID[i].item():tokenizer.decode(torch.abs(I.T[i])[torch.nonzero(I.T[i],as_tuple=True)].long().tolist())}
#part=partial(createdict, I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)

def createRESFile(LSTM,VQA,resLocation):
    #def CreateResFileFromModelpt(resFile,vqa,modellocation=os.path.join(modelDir,"modelVitContrastive2epoch9.pt")):
    results=[]
    Batchsize=300
    data=myVQALSTM(VQA, model, tokenizer, preprocess,train=False)
    dataloader=torch.utils.data.DataLoader(data,
                                          batch_size=Batchsize, 
                                          num_workers=8,
                                          shuffle=False,
                                          prefetch_factor=2,
                                          drop_last=False,
                                          pin_memory=True,
                               )
    data=tqdm(dataloader)
    with torch.no_grad():
        for images,Q,QA,A,QID in data:
            image_input=images.cuda()
            Q=Q.cuda()#to(device,non_blocking=True)#Bx77
            #QA=QA.cuda()#to(device,non_blocking=True)#Bx77
            #image_input -= image_mean[:, None, None]
            #image_input /= image_std[:, None, None]
            #GTAnswers=A.cuda() #Bx767
            #outs=torch.zeros((A.size(0),A.size(1),).cuda()
            _,startindexes=torch.max(Q==0,dim=1)#B
            #print(startindexes.size())#B
            #optimizer.zero_grad()
            startindexes[startindexes<=0]=0
            #print(Answers)
            outputs=LSTM(image_input,Q.T,torch.zeros(Q.T.size()))#.permute(1,0,2)#[77, B, 49408]) to B,77,V
            #outs=torch.zeros(outputs.size()).cuda()
            make_dot(LSTM(image_input,Q.T,torch.zeros(Q.T.size())), params=dict(list(LSTM.named_parameters())),  show_attrs=True, show_saved=True)

            del Q,QA,image_input
            #print(outputs.size())#B,77,V
            #insert=torch.zeros(B,77)
        
            _,I=torch.max(outputs,dim=-1)
            part=partial(createdict, I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)

            #Answers[:77-startindexes[i],i]=QA.T[startindexes[i]:,i].detach()
            with Pool(20) as P:
                lis=P.map(part,range(startindexes.size(0)))
            results.extend(lis)
            del I,outputs,lis
#             for i in range(startindexes.size(0)):
#                 #try:
#                 #indexs=torch.arange(startindexes[i],77).cuda()
#                 #outputs[:,i]=F.pad(torch.index_select(outputs[:,i],0,indexs.long()),(0,0,0,startindexes[i]))
#                 text=tokenizer.decode(torch.abs(I.T[i]).long().tolist())
#                 print(text)
#                 results.append({QID[i].item():text})
                #except:
                #    print(startindexes[i])
                    #outputs[:,i]=torch.index_select(outputs[:,i],0,indexs.long())
            #results.extend(dict(zip(QID,tokenizer.decode(outputs))))
            #print(tokenizer.decode(outputs))
    json.dump(results,  open(resLocation,  'w'))


# -

LSTM=torch.load('CLIPLSTMv3efin.pt').eval()
LSTM.teacher_forcing_ratio=0
createRESFile(LSTM,vqa,"testdevCLIPLSTMRESv3.json")

LSTM=torch.load('CLIPLSTMv4e6.pt').eval()
LSTM.teacher_forcing_ratio=0
createRESFile(LSTM,vqa,"testdevCLIPLSTMRESv4.json")

LSTM=torch.load('CLIPLSTM2.pt').eval()
LSTM.teacher_forcing_ratio=0
createRESFile(LSTM,vqa,"testdevCLIPLSTMRESv0.json")

LSTM=torch.load('CLIPLSTMv2e116.pt').eval()
LSTM.teacher_forcing_ratio=0
createRESFile(LSTM,vqa,"testdevCLIPLSTMRESv2.json")

LSTM=torch.load('./data/models/CLOZELSTMs/CLIPLSTMv5e6.pt').eval()
LSTM.teacher_forcing_ratio=0
createRESFile(LSTM,vqa,"testCLIPLSTMRESv5e6.json")

tokenizer.encode("Word for word test or char to char")

dataloader._get_iterator().next()__dir__()


# +
data=myVQALSTM(vqa, model, tokenizer, preprocess,train=False)
dataloader=torch.utils.data.DataLoader(data,
                                          batch_size=2, 
                                          num_workers=8,
                                          shuffle=False,
                                          prefetch_factor=2,
                                          drop_last=False,
                                          pin_memory=True,
                               )
LSTM=torch.load('./data/models/CLOZELSTMs/CLIPLSTMv5e6.pt').eval()
LSTM.device="cpu"
from torchviz import make_dot, make_dot_from_trace
device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

def patch_device(module):
    graphs = [module.graph] if hasattr(module, "graph") else []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for graph in graphs:
        for node in graph.findAllNodes("prim::Constant"):
            if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                node.copyAttributes(device_node)

LSTM.apply(patch_device)
patch_device(LSTM)

# patch dtype to float32 on CPU
##if device == "cpu":
float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
float_node = float_input.node()

def patch_float(module):
    graphs = [module.graph] if hasattr(module, "graph") else []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for graph in graphs:
        for node in graph.findAllNodes("aten::to"):
            inputs = list(node.inputs())
            for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                if inputs[i].node()["value"] == 5:
                    inputs[i].node().copyAttributes(float_node)
LSTM.apply(patch_float)
LSTM=LSTM.eval()
LSTM.teacher_forcing_ratio=0
images,Q,_,_,_=dataloader._get_iterator().next()
LSTM=LSTM.float().cpu()
#make_dot(LSTM(images.cpu(),Q.T.cpu(),torch.zeros(Q.T.size()).cpu()).float(), params=dict(list(LSTM.named_parameters())),  show_attrs=True, show_saved=True)

# -

CreateResFileFromModelpt("test2k15.json",vqa,modellocation=os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt"))


class psCLIPLSTM(CLIPLSTM):
    def __init__(self,clip,dropout_p=0.1,max_length=77,teacher_forcing_ratio=0.5,Batch=100):   

        super().__init__(clip,dropout_p=dropout_p,max_length=max_length,teacher_forcing_ratio=teacher_forcing_ratio)
        self.lstm=nn.lstm(input_size=Batch,hidden_size=self.hidden_size,num_layers=1,bidirectional=True)
     def forward(self, image_tensor, QTensor,GTQA_tensor): # 
        Batch= image_tensor.size(0)
        encoder_output = self.clip.encode_image(image_tensor).float()
        encoder_output = encoder_output / encoder_output.norm(dim=-1, keepdim=True)
        encoder_hidden = self.clip.encode_text(QTensor.T).float()
        encoder_hidden = encoder_hidden / encoder_hidden.norm(dim=-1, keepdim=True)
        #decoder_input=QTensor[0] #symbols B

        return self.lstm(QTensor[0],(encoder_hidden.float(),encoder_output))


m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
for i in range(1000):
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output.backward()
    print(output.item())


def rotatebyA(Source,AmounttoRotate):
    V,I=torch.max(AmounttoRotate,dim=0)
    pad=torch.zeros((V.long().item(),Source.size(1)))
    Source=torch.cat((Source,pad),dim=1)
    out=torch.stack( [torch.roll(Source[i],-AmounttoRotate[i].long().item(),0) for i in range(Source.size(0))])
    return out.T[:V.long().item()].T

