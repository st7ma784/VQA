import torch 
from simple_tokenizer import SimpleTokenizer
from pycocotools.coco import COCO
import torchvision.datasets as dset
import torch
import torchvision
import torchnlp
from torchnlp.samplers.noisy_sorted_sampler import NoisySortedSampler
import numpy as np 
import random
import sys
from pathlib import Path
B=384
from tqdm import tqdm 
from sklearn.cluster import MiniBatchKMeans
import torchvision.transforms as T
from utils import box_cxcywh_to_xyxy,rescale_bboxes,getArea
class mybothCocoCaptions(dset.CocoCaptions):
    def __init__(
            self,
            root: str,
            annFile: str,
            transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ]),
            target_transform = None,
            transforms = None,
            context_length=300
    ):
        super(mybothCocoCaptions, self).__init__(root, annFile, transform,target_transform,transforms)
        self.context_length=context_length
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']

    def __getitem__(self, index: int):
        """
        Args:
        index (int): Index

        Returns:
        tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img,target= super().__getitem__(index)
        #target = [ann['caption'] for ann in anns]#
        #print(target)
        text= random.choice(target) 


        text_tokens = [self.sot_token] + self.tokenizer.encode(text) + [self.eot_token]
        #print(text_tokens)
        text_input = torch.zeros(self.context_length, dtype=torch.long)
        text_input[:len(text_tokens)] = torch.tensor(text_tokens)
        return  text_input,img

class myCocoCaptions(dset.CocoCaptions):
    def __init__(
            self,
            root='train2017',
            annFile='{}/annotations/captions_{}.json'.format('.','train2017'),
            transform = None,
            target_transform = None,
            transforms = None,
            context_length=300,

    ):
        super(myCocoCaptions, self).__init__(root, annFile, transform,target_transform,transforms)
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.context_length=context_length
    def __getitem__(self, index: int):
        """
        Args:
        index (int): Index

        Returns:
        tuple: Tuple (image, target). target is a list of captions for the image.
        """
        _,target= super().__getitem__(index)
        #target = [ann['caption'] for ann in anns]#
        #print(target)
        text= random.choice(target) 
        text_tokens = [self.sot_token] + self.tokenizer.encode(text) + [self.eot_token]
        #print(text_tokens)
        text_input = torch.zeros(self.context_length, dtype=torch.long)
        text_input[:len(text_tokens)] = torch.LongTensor(text_tokens)
        return text_input