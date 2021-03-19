from torch.utils.data.sampler import Sampler
from torchnlp.utils import identity
import random
from torch.utils.data.sampler import BatchSampler
from utils import categorisor
from torch.utils.data import DataLoader
import itertools
import numpy as np
import torch
import random
import os
import sys
from pathlib import Path
import pickle
from tqdm import tqdm 
from sklearn.cluster import MiniBatchKMeans
from torch.autograd import Variable
from concurrent.futures import ProcessPoolExecutor, wait,ALL_COMPLETED

def getSortedSampler(dataset4sort,model,device,B):
    pkl_filename = os.path.join(os.getcwd(),"samplers/pickle_batch_sorted_b{}_sampler.pkl".format(B))

    # try:
    #     print("Trying to open Sampler from : {}".format(pkl_filename))
    #     with open(pkl_filename, 'rb') as file:
    #         train_sampler2 = pickle.load(file)
    #     print("Opened sampler from file...")
    # except:
    get_noise = lambda i: round(random.uniform(-1, 1))
    
    train_sampler2=NoisySortedBatchSampler(
                    data=dataset4sort,
                    model=model,
                    device=device,
                    batch_size=B,
                    drop_last=True,
                    shuffle=True)
    print("Saving dataLoader to {}".format(pkl_filename))
    with open(pkl_filename, 'wb') as file:
        pickle.dump(train_sampler2, file)
    return train_sampler2

def sortTexts(Texts,model,device):
    data=DataLoader(Texts,batch_size=200,num_workers=4,pin_memory=True)
    #device='cpu'
    model.to(device)
    with torch.no_grad(): 

        pkl_filename = "samplers/pickle_kmeans_model.pkl"
        try:
            with open(pkl_filename, 'rb') as file:
                modelkmeans = pickle.load(file)
            print("Opened KMeans from file...")
        except:
            modelkmeans = MiniBatchKMeans(n_clusters=100, init='k-means++', n_init=10, batch_size=200)
            for i in tqdm(data):

                text_features = model.encode_text(i.to(device)).float().detach().cpu()
                modelkmeans.partial_fit(text_features)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(modelkmeans, file)

        # could remove and use detach later?   
        results=[]
        for i in tqdm(data):
            results.append(modelkmeans.predict(model.encode_text(i.to(device)).float().detach().cpu()))

        return np.concatenate(results)
     
class SortedSampler(Sampler):

    def __init__(self,
                 data,
                 model,
                 device,
                 get_noise= lambda i: 0,
                 sort_key=identity,
                 ):
        super().__init__(data)
        self.data = data
        self.model=model
        self.device=device
        noise=np.asarray([get_noise(r) for r in range(len(self.data))])
        order = sortTexts(self.data,self.model,self.device)
        zip_= enumerate(order+noise)
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)



class NoisySortedBatchSampler(BatchSampler):


    def __init__(self,
                 data,
                 model,
                 device,
                 batch_size,
                 drop_last,
                 shuffle=True):
        self.shuffle = shuffle
        self.model=model
        self.device=device
        super().__init__(
            SortedSampler(data=data, model=self.model,device=self.device),
            batch_size, drop_last)

    def __iter__(self):
        batches = list(super().__iter__())

        if self.shuffle:
            random.shuffle(batches)
 
        return iter(batches)
