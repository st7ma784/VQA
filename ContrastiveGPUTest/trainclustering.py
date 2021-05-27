from pycocotools.coco import COCO
import torchvision.datasets as dset
import torch
import torchvision
import torchnlp
from pathlib import Path
import numpy as np 
import random
import os
import sys
import pickle
from tqdm import tqdm 

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT,"CLIP")))

#from torchnlp.samplers.noisy_sorted_batch_sampler import NoisySortedBatchSampler
#sdir=os.path.dirname(os.path.realpath(os.path.join(os.getcwd(),".." )))
# clip_dir =os.path.realpath(os.path.join("..","..","CLIP"))
# print(f"CLIP dir is: {clip_dir.absolute}")
# sys.path.append(str(clip_dir))

import clip
from samplers.mySampler import NoisySortedBatchSampler,SortedSampler,getSortedSampler

from sklearn.cluster import MiniBatchKMeans
import torchvision.transforms as T
from torch.autograd import Variable
from utils import AverageMeter,categorisor
from utils import box_cxcywh_to_xyxy,rescale_bboxes,getArea

from model import convert_weights
from datasets.captionloader import mybothCocoCaptions
from datasets.captionloader import myCocoCaptions

B=384
device = 'cuda:0'
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("...of which trainable:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters() if p.requires_grad]):,}")
embed_size= model.state_dict()["text_projection"].shape[1]
print("Embed size:", embed_size)
mtransform=T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                ]),
dataDir='data'
folders=['train2017','train2014']

def train(model, optimizer, train_loader, device, scheduler=None,):
    model.to(device).train()
    tk0 = tqdm(train_loader, total=len(train_loader))
    summary_loss = AverageMeter()

    labels = torch.LongTensor(np.arange(B)).to(device)
    torch.autograd.set_detect_anomaly(True)
    for data in tk0:
        texts,images=data
        #could average text inputs
        image_input =Variable(images, requires_grad=True).to(device,non_blocking=True)
        text_input = Variable(texts).to(device,non_blocking=True)
        criterion = torch.nn.CrossEntropyLoss().to(device,non_blocking=True)
        criterion2 = torch.nn.CrossEntropyLoss().to(device,non_blocking=True)
        optimizer.zero_grad()
        #subimages=combineSubImages(getSubImages(images,device2=device),model,device)
        logitsi,logitst= model.forward(image_input,text_input)# logits_per_iamge, logits_per_text
        del texts,images,text_input,image_input
        loss_i=criterion(logitsi,labels)    #start_logits.view(-1, 2)
        loss_t=criterion2(logitst,labels)
        loss=(loss_i+loss_t)/2
        torch.autograd.backward(loss)
        #print("sorting grads...")
        del loss_i,loss_t
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        summary_loss.update(loss.item())
        del loss
        tk0.set_postfix(loss=summary_loss.avg)
    return summary_loss.avg

def train_fn(data_loaders,net,epochs=10, lr=0.005):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, eps= 1e-2)
    for e in range(epochs): 
        for loader in data_loaders:
            loss=train(net, optimizer, loader, device)
            #print(loss)
            #save model here. 
        if (e+1)%10 ==0:
            torch.save(net.state_dict(), "./data/models/model{}epoch{}.pt".format("VitContrastive2",e))
def getDataloaders(folders,model):
    dataloaders=[]
    for folder in folders:
        dataType=folder#'train2017'
        annFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
        dataset4sort=myCocoCaptions(root = os.path.join(dataDir,dataType),
                                annFile = annFile,
                                context_length=model.context_length,
                                )
        train_sampler2=getSortedSampler(dataset4sort,model.encode_text,device,B)
        dataset=mybothCocoCaptions(root = os.path.join(dataDir,dataType),
                                    annFile = annFile,
                                    context_length=model.context_length,
                                )
        dataloaders.append(torch.utils.data.DataLoader(dataset,
                                          #batch_size=350, 
                                          num_workers=4,
                                          #shuffle=True,
                                          #sampler=train_sampler,
                                          batch_sampler=train_sampler2,
                                          prefetch_factor=3,
                                          #drop_last=True,
                                          pin_memory=True,
                                          )
                            )
    return dataloaders
#print("here,  consider saving depending on how long this takes? ")
LR=0.00005
data_loaders=getDataloaders(folders,model)
train_fn(data_loaders,model,epochs=100,lr=LR)

