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

B=100
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
dataType='train2017'
annFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
def getSubImages(im,detrmodel,model,device2=device):
        #To do: check images arent full size bboxes
        outputs = detrmodel(im)
        e=torch.nn.EmbeddingBag(100, model.visual.output_dim,mode='mean').to(device,non_blocking=True)
        pred_boxes=outputs['pred_boxes']       #100,100,4
        x1,y1,W,H=torch.unbind(pred_boxes,2)  #in cxywh     4x [100,100]
        affineMatrixX=torch.stack([W,torch.zeros(x1.shape,device=device),x1],dim=-1)   #is shape  100,100,3
        affineMatrixY=torch.stack([torch.zeros(x1.shape,device=device), H,y1],dim=-1)
        theta=torch.stack([affineMatrixX,affineMatrixY],dim=-2) #100,100,2,3
        #need SANITY CHECK HERE
        logits=[]
        #imThetas=torch.unbind(theta,0)
        for image,theta in zip(torch.unbind(im,0),torch.unbind(theta,0)): #for [3,224,224], [100,2,3] in zip
            #print(im.shape)
            #print(theta.shape)
            affinegrid=torch.nn.functional.affine_grid(theta, torch.Size((100,3,224,224)),align_corners=True)
            new=torch.nn.functional.grid_sample(torch.stack(100*[image],dim=0), affinegrid,align_corners=True)
            output=model.encode_image(new).long()
            #print(output.shape)
            trans = transforms.ToPILImage()
            plt.imshow(trans(img))


            logit=e(output)[-1]
            #print(logit.shape)
            logits.append(logit)
        return torch.stack(logits,dim=0)


def train(model, optimizer, train_loader, device, scheduler=None,):
    detrmodel=torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).train().cuda()
    labels = torch.LongTensor(np.arange(B)).to(device)
    torch.autograd.set_detect_anomaly(True)
    for data in tk0:
        texts,images=data
        #could average text inputs
        inputs=torch.Tensor(images).to(device,non_blocking=True)

        subim_features=getSubImages(inputs,detrmodel,model)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        subim_features = subim_features / subim_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return summary_loss.avg

def train_fn(data_loader,net,epochs=10, lr=0.005):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, eps= 1e-2)
    for e in range(epochs): 
        loss=train(net, optimizer, data_loader, device)
        #print(loss)
        #save model here. 
        if e+1 %10 ==0:
            torch.save(net.state_dict(), "./model{}epoch{}.pt".format("VitSubimages",e))


dataset4sort=myCocoCaptions(root = os.path.join(dataDir,dataType),
                            annFile = annFile,
                            context_length=model.context_length,
                            )

train_sampler2=getSortedSampler(dataset4sort,model,device,B)
dataset=mybothCocoCaptions(root = os.path.join(dataDir,dataType),
                            annFile = annFile,
                            context_length=model.context_length,
                            )
data_loader = torch.utils.data.DataLoader(dataset,
                                          #batch_size=350, 
                                          num_workers=4,
                                          #shuffle=True,
                                          #sampler=train_sampler,
                                          batch_sampler=train_sampler2,
                                          #prefetch_factor=2,
                                          #drop_last=True,
                                          pin_memory=True)
#print("here,  consider saving depending on how long this takes? ")
LR=0.00005

train_fn(data_loader,model,epochs=100,lr=LR)



# +
import torch
def rotatebyA(Source,AmounttoRotate):
    V,I=torch.max(AmounttoRotate,dim=0)
    pad=torch.zeros(Source.size())[:V.long().item()]
    Source=torch.cat((Source,pad),dim=1)
    out=torch.stack( [torch.roll(Source[i],-AmounttoRotate[i].long().item(),0) for i in range(Source.size(0))])
    return out[:,:V.long().item()]

test=torch.range(0,99).view(10,10)
#out=rotatebyA(test,torch.range(0,10))
#print(out)
out=list()
for i in range(0,10):
    out.append(torch.tensor(test[i][test.nonzero()]))
    print(torch.sum(test[i]!=0))#test[i][test[i].nonzero()])
noz=test[test.nonzero()]
# +
import torchnlp.encoders.text as Tex


stack=Tex.stack_and_pad_tensors(noz,0,10)
print(stack.tensor.size())
print(torch.max(stack.lengths,dim=-1))
# -

print(stack)



