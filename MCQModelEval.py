
import json
import datetime
import copy
import re
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os,sys
import torch.nn as nn
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGE_PARENT = '..'
SCRIPT_DIR = "."
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "CLIP")))
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
from multiprocessing import Pool
from clozeGeneration.LSTM import CLIPLSTM,swCLIPLSTM
from clozeGeneration.utils import AverageMeter
from CLIP.clip.simple_tokenizer import SimpleTokenizer
import CLIP.clip as CLIP
from collections import ChainMap
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from VQA.PythonHelperTools.vqaTools.vqa import *
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import *
import random
import torch.nn as nn
import torch
from torch import optim

def patch_device(module):
    graphs = [module.graph] if hasattr(module, "graph") else []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for graph in graphs:
        for node in graph.findAllNodes("prim::Constant"):
            if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                node.copyAttributes(device_node)

from torch.utils.tensorboard import SummaryWriter

def convertQA(vqa,question_id,ans):
    #print(vqa.qqa[question_id])
    text=vqa.qqa[question_id]['question'][:-1] #question qa
    return " ".join([text,ans])

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
            self.lookup=dict((i,j) for i,j in enumerate(self.dicts))
            self.candidateslist={ann['question_id']:list(set(ans["answer"] for ans in ann['answers'])) for ann in anns}
            for ann in anns:
                quesId = ann['question_id']
                print( "Question: %s" %(self.qqa[quesId]['question']))
                print(list(set(ans["answer"] for ans in ann['answers'])))   
    def loadIMG(self,imgFilename):
        return self.preprocess(Image.open(os.path.join(data_dir, imgFilename)).convert("RGB"))
    def getFilename(self,qid,):
        return 'COCO_' + dataSubType + '_'+ str(self.vqa.qqa[qid]['image_id']).zfill(12) + '.jpg'
    def loadTextTensors(self,qid):
        qid=int(qid)
        candidates=self.candidateslist[qid]#[dic[qid] for dic in self.dicts if qid in dic]
        candidateQA=[convertQA(self.vqa,qid,candidate) for candidate in candidates]
        #process in some way?? 
       
        text_tokens = [self.tokenizer.encode(desc) for desc in candidateQA]
        text_input = torch.zeros((len(self.dicts), self.model.context_length), dtype=torch.long)
        for i, tokens in enumerate(text_tokens):
            tokens = [self.tokenizer.encoder['<|startoftext|>']] + tokens
            text_input[i, :len(tokens)] = torch.tensor(tokens)  
        text_tokens.append(torch.zeros(tuple([self.model.context_length])))
        for i in range(len(text_tokens),12):
            text_input[i,:]=torch.randint(4,49000,tuple([self.model.context_length]))
            text_input[i,0]=torch.tensor([self.tokenizer.encoder['<|startoftext|>']])
            text_input[i,-1]=torch.tensor([self.tokenizer.encoder['<|endoftext|>']])
        return len(candidates),text_input
    def __len__(self):
        return len(self.qids)
    def __getitem__(self, index: int):
        qid=int(self.qids[index])
        filename=self.getFilename(qid)
        im=self.loadIMG(filename)
        lencandidates,textinputs=self.loadTextTensors(qid)
        return im,textinputs,lencandidates,qid

def createdict(i,similarity=[],qids=[],data=[],tokenizer=SimpleTokenizer()):
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
    '''Given Resfile location, save there the best MCQA from data and model'''
    res=[]
    Batchsize=2000
    dataloader=torch.utils.data.DataLoader(data,
                                          batch_size=Batchsize, 
                                          num_workers=10,
                                          shuffle=True,
                                          prefetch_factor=2,
                                          drop_last=False,
                                          pin_memory=True,
                               )
    dudQs=0
    dudQsIDS=torch.tensor()
    with torch.no_grad():
        for images,textinputs,answerlistlengths,qids in tqdm(dataloader):
            text_input=textinputs.cuda()
            image_input=images.cuda()
            image_input -= image_mean[:, None, None]
            image_input /= image_std[:, None, None]
            
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            slices=[]
            for i in range(text_input.size(1)):
                textslice=model.encode_text(text_input[:,i]).float()
                textslice /= textslice.norm(dim=-1, keepdim=True)
                slices.append(torch.diagonal(textslice @ image_features.T))
            similarity=torch.stack(slices,dim=-1)
            Values,Indices=torch.max(similarity,dim=-1)
            count=torch.sum(Indices>answerlistlengths)
            dudQs+=count.item()
            dudQsIDS=torch.cat((dudQsIDS,Indices[Indices>answerlistlengths]))
            part=partial(createdict,similarity=similarity.cpu(),qids=qids.long().detach().cpu(),data=data.candidateslist) #I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)
            with Pool(20) as P:
                res.extend(list(P.map(part,range(qids.size(0)))))
    json.dump(res,  open(resFile,  'w'))
    print("DUD Qs Count: {}".format(dudQs))
    return dudQs,dudQsIDS
    
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
    


def createdict(i,I=[],QID=[],tokenizer=[]):
    
    
    return {QID[i].item():tokenizer.decode(torch.abs(I.T[i])[torch.nonzero(I.T[i],as_tuple=True)].long().tolist())}
#part=partial(createdict, I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)

def createRESFile(LSTM,VQA,resLocation):
    results=[]
    Batchsize=900
    data=myVQALSTM(VQA, model, tokenizer, preprocess,train=False)
    dataloader=torch.utils.data.DataLoader(data,
                                          batch_size=Batchsize, 
                                          num_workers=8,
                                          shuffle=True,
                                          prefetch_factor=2,
                                          drop_last=False,
                                          pin_memory=True,
                               )
    data=tqdm(dataloader)
    with torch.no_grad():
        for images,Q,_,_,QID in data:
            image_input=images.cuda()
            Q=Q.cuda()#to(device,non_blocking=True)#Bx77
            _,startindexes=torch.max(Q==0,dim=1)#B
            startindexes[startindexes<=0]=0
            outputs=LSTM(image_input,Q.T)#,torch.zeros(Q.T.size()))#.permute(1,0,2)#[77, B, 49408]) to B,77,V
            #make_dot(LSTM(image_input,Q.T,torch.zeros(Q.T.size())), params=dict(list(LSTM.named_parameters())),  show_attrs=True, show_saved=True)
            #del Q,image_input
            _,I=torch.max(outputs,dim=-1)
            part=partial(createdict, I=I.cpu(),QID=QID.cpu(),tokenizer=tokenizer)
            with Pool(20) as P:
                lis=P.map(part,range(startindexes.size(0)))
            results.extend(lis)
            del I,outputs,lis
    json.dump(results,  open(resLocation,  'w'))

if __name__ == '__main__':
        
    model, preprocess = CLIP.load("ViT-B/32", device=device, jit=True)
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    #print("Failed JitLoad")
    model, _ = CLIP.load("ViT-B/32", device=device, jit=False)

    # %ls
    # +
    dataDir		='./data'#'../../VQA'
    versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
    dataSubType ='train2014'
    annFile     ='%s/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
    quesFile    ='%s/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
    imgDir 		= '%s/%s/' %(dataDir, dataSubType)
    fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
    data_dir=os.path.join(dataDir,dataSubType)

    [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
    fileType) for fileType in fileTypes]  

    #vqa=VQA(question_file= quesFile)
    vqa=VQA(annotation_file=annFile,question_file= quesFile)
    tokenizer= SimpleTokenizer()
    modelDir="./data/models/"
    model.load_state_dict(torch.load(os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt")))
    model.cuda()
    data=myVQALoader(vqa, model,tokenizer, preprocess)
    #data=myVQALoader(vqa, model, tokenizer, preprocess)
    Resname="MCQResults.json"

    count,dudQIDS=CreateResFileFromModelpt(Resname,data,model)
    print(dudQIDS)







    # torch.autograd.set_detect_anomaly(True)
    # Batchsize=120

    # writer = SummaryWriter('runs/LSTM_experiment_1')
    # LSTM= psCLIPLSTM(model).float().cuda() #torch.load('./data/models/CLOZELSTMs/CLIPLSTMv5e6.pt')
    # LSTM.device=device
    # #model.apply(patch_device)
    # #LSTM.apply(patch_device)
    # #patch_device(LSTM)
    # #patch_device(model)
    # name='./data/models/CLOZELSTMs/CLIPLSTMv7e59.pt'
    # training=False
    # if training: 
    #     data=myVQALSTM(VQA(annotation_file=annFile,question_file= quesFile), model, tokenizer, preprocess)
    #     dataloader=torch.utils.data.DataLoader(data,
    #                                         batch_size=Batchsize, 
    #                                         num_workers=6,
    #                                         shuffle=True,
    #                                         prefetch_factor=2,
    #                                         drop_last=False,
    #                                         pin_memory=True,
    #                             )
    #     optimizer=torch.optim.AdamW([p for p in LSTM.parameters() if p.requires_grad], lr=0.00003,eps= 1e-3)
    #     name=train(LSTM,dataloader, 100,optimizer) 
    #     #torch.save(LSTM, "./data/models/CLOZELSTMs/CLIPLSTMv6e7.pt")
    # LSTM=torch.load(name).eval()
    # try:
    #     LSTM.teacher_forcing_ratio=0
    # except:
    #     pass
    # Resname="train2k14CLIPLSTMRESv7e59.json"
    # createRESFile(LSTM,vqa,Resname)

    # with open(Resname) as f:
    #             #print(lis)
    #     word_pattern = re.compile('<\|endoftext\|>')
    #     new=list({"question_id":int(key), "answer":word_pattern.sub("",val)} for k in list(json.loads(f.read())) for key, val in k.items())
        


    #     json.dump(new,  open(Resname,  'w'))
    # EvaluateResFile(annFile,quesFile,Resname)
    #createRESFile(LSTM,vqa,"testCLIPLSTMRESv5e6.json")


    # device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    # device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    # LSTM.apply(patch_device)
    # #patch_device(LSTM.encode_image)
    # #patch_device(LSTM.encode_text)

    # # patch dtype to float32 on CPU
    # if device == "cpu":
    #     float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    #     float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
    #     float_node = float_input.node()

    #     def patch_float(module):
    #         graphs = [module.graph] if hasattr(module, "graph") else []
    #         if hasattr(module, "forward1"):
    #             graphs.append(module.forward1.graph)

    #         for graph in graphs:
    #             for node in graph.findAllNodes("aten::to"):
    #                 inputs = list(node.inputs())
    #                 for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
    #                     if inputs[i].node()["value"] == 5:
    #                         inputs[i].node().copyAttributes(float_node)
    #     model.apply(patch_float)
    #     patch_float(model.encode_image)
    #     patch_float(model.encode_text)



    # LSTM.teacher_forcing_ratio=0


    # data=myVQALSTM(vqa, model, tokenizer, preprocess,train=False)
    # dataloader=torch.utils.data.DataLoader(data,
    #                                         batch_size=2, 
    #                                         num_workers=8,
    #                                         shuffle=False,
    #                                         prefetch_factor=2,
    #                                         drop_last=False,
    #                                         pin_memory=True,
    #                             )
    # LSTM=torch.load('./data/models/CLOZELSTMs/CLIPLSTMv5e6.pt').eval()
    # LSTM.device="cpu"

    # LSTM.apply(patch_device)
    # patch_device(LSTM)

    # # patch dtype to float32 on CPU
    # ##if device == "cpu":
    # float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    # float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
    # float_node = float_input.node()

    # def patch_float(module):
    #     graphs = [module.graph] if hasattr(module, "graph") else []
    #     if hasattr(module, "forward1"):
    #         graphs.append(module.forward1.graph)

    #     for graph in graphs:
    #         for node in graph.findAllNodes("aten::to"):
    #             inputs = list(node.inputs())
    #             for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
    #                 if inputs[i].node()["value"] == 5:
    #                     inputs[i].node().copyAttributes(float_node)
    # LSTM.apply(patch_float)
    # LSTM=LSTM.eval()
    # LSTM.teacher_forcing_ratio=0
    # images,Q,_,_,_=dataloader._get_iterator().next()
    # LSTM=LSTM.float().cpu()





    #CreateResFileFromModelpt("test2k15.json",data,model)#,modellocation=os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt"))


    #vqa=VQA(annotation_file=annFile,question_file= quesFile)

    #CreateResFileFromModelpt(resFile,vqa,modellocation=os.path.join(modelDir,"modelVitUpdatingContrastive2epoch59.pt"))
    #EvaluateResFile(annFile,quesFile,resFile,)

# for i,qid in enumerate(random.choices(vqa.getQuesIds(),k=8)):
#     question=vqa.loadQA(qid)[0]
#     ImageId=question['image_id']
#     text=vqa.qqa[qid]['question'][:-1]
#     candidates=list(set(ans["answer"] for ans in question['answers']))
#     candidateQA=[" ".join([text,candidate]) for candidate in candidates]
#     imgFilename = 'COCO_' + dataSubType + '_'+ str(ImageId).zfill(12) + '.jpg'
#     images=[]
#     im=Image.open(os.path.join(data_dir, imgFilename)).convert("RGB")
#     image = preprocess(im)
#     images.append(image)
#     image_input = torch.tensor(np.stack(images)).cuda()
#     image_input -= image_mean[:, None, None]
#     image_input /= image_std[:, None, None]
#     text_tokens = [tokenizer.encode(desc) for desc in candidateQA]
#     text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
#     for i, tokens in enumerate(text_tokens):
#         tokens = [tokenizer.encoder['<|startoftext|>']] + tokens + [ tokenizer.encoder['<|endoftext|>']]
#         text_input[i, :len(tokens)] = torch.tensor(tokens)

#     text_input = text_input.cuda()
#     with torch.no_grad():
#         image_features = model.encode_image(image_input).float()
#         text_features = model.encode_text(text_input).float()
#     #image_features /= image_features.norm(dim=-1, keepdim=True)
#     #text_features /= text_features.norm(dim=-1, keepdim=True)

#     similarity = text_features.cpu() @ image_features.cpu().T

#     similarity= similarity.T#F.normalize(similarity.T, p=2,)
#     Values,Indices=torch.max(similarity,dim=-1)

#     #print(Indices)
#     top_probs, top_labels = similarity.cpu().topk(len(candidates))
#     print("Q:{} \n A:{}".format(text,candidates[Indices.item()]))
#     plt.subplot(4, 4, 2 * i +1)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.axis("off")

#     plt.subplot(4, 4, 2 * i + 2)
#     y = np.arange(top_probs.shape[-1])
#     plt.grid()
#     print(top_probs)
#     plt.barh(y, top_probs.T)
#     plt.gca().invert_yaxis()
#     #plt.gca().set_axisbelow(True)
#     plt.yticks(y, [candidates[top_labels.T[i].item()] for i in range(len(candidates))])
#     plt.xlabel(text)
#     #resfile= [{"answer":str, "question_id":int}]
# plt.subplots_adjust(wspace=0.5)

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




