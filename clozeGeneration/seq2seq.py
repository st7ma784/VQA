from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *
from LSTM import *
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
teacher_forcing_ratio=0.5
dir="./data/data/asnq"
Batch=900#2500

# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=20):
#     Batch= input_tensor.size(1)
#     encoder_hidden =torch.zeros(1, Batch, encoder.hidden_size, device=device)
#  #shape [1,B,256]
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()

#     input_length = input_tensor.size(0)   #20
#     target_length = target_tensor.size(0) #20
#     loss = 0

#     encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)# [20,b,256],[1,b,256]
#     encoder_outputs = torch.zeros(encoder_output.shape, device=device)#20,b,256
#     #print("Encoder Output: {}".format(encoder_output.shape))# [20, b,256]
#     #print("Encoder Hidden: {}".format(encoder_hidden.shape))# [1,b,256]

#     #for ei in range(input_length):
#     encoder_outputs[:] = encoder_output[0,:,:]
#     #print("Encoder Outputs: {}".format(encoder_outputs.shape))# [20, b,256]

#     encoder_outputs=encoder_outputs.permute(1,0,2) #[b,20,256]
#     decoder_input = torch.tensor([SOT]*Batch, device=device) #[B]
#     #print("decoder in : {}".format(decoder_input.shape)) #[b]
#     decoder_hidden = encoder_hidden
#     #print(decoder_hidden.shape) #[1,B,256]

#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden[0], encoder_outputs)
#             #[B,n_words] =#decoder([b], [b,1,256] , [20,b,256])
#             #print("Decoder hidden : {}".format(decoder_hidden.shape)) 
#             #print("Decoder attn : {}".format(decoder_attention.shape)) 
#             #print("Decoder out : {}".format(decoder_output.shape)) 

#             #print(target_tensor.shape)#20,7
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di] # Teacher forcing [B]
#             #print("decoder in : {}".format(decoder_input.shape)) #[b]

#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden[0], encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input
#             print(dataloader.tokenizer.decode([decoder_input[0].item()]))
#             loss += criterion(decoder_output, target_tensor[di])
#             # if decoder_input.item() == EOT:
#             #     break

#     loss.backward()

#     encoder_optimizer.step()
#     decoder_optimizer.step()

#     return loss.item() / target_length


# def trainIters(encoder, decoder, n_iters, learning_rate=0.0001):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every

#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     # training_pairs = [pair for pair in dataloader]
#     criterion = nn.NLLLoss()
#     for i in range(n_iters):
#         summary_loss = AverageMeter()
#         data=tqdm(data_loader)
#         for Train,Test in data:
#             input_tensor = Train.T.to(device,non_blocking=True)
#             target_tensor = Test.T.to(device,non_blocking=True)
#             #print(input_tensor.size())
#             #print(target_tensor.size())
#             loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#             summary_loss.update(loss)
#             data.set_postfix(loss=summary_loss.avg)



            # loss += criterion(decoder_output, target_tensor[di])

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np


# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)

# def evaluate(input_tensor,target_tensor,encoder, decoder, max_length=20):
#     with torch.no_grad():
#         Batch= input_tensor.size(1)
#         encoder_hidden =torch.zeros(1, Batch, encoder.hidden_size, device=device)
#         input_length = max_length
#         target_length = max_length
#         loss = 0

#         encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)# [20,b,256],[1,b,256]
#         encoder_outputs = torch.zeros(encoder_output.shape, device=device)#20,b,256
#         encoder_outputs[:] = encoder_output[0,:,:]
#         encoder_outputs=encoder_outputs.permute(1,0,2) #[b,20,256]
#         decoder_input = torch.tensor([SOT]*Batch, device=device) #[B]
#         decoder_hidden = encoder_hidden
#         decoded_words=[]
#         decoder_attentions = torch.zeros(max_length, max_length)
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden[0], encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             # print("decoder input : {}".format(topi))
#             # decoder_attentions[di] = decoder_attention.data
#             decoder_input = topi  # detach from history as input
#             decoded_words.append(topi.item())

#         return decoded_words, decoder_attentions[:di + 1]
# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')


#sshare embedding space between these? 

# encoder1 = EncoderRNN(n_words, hidden_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size,n_words, dropout_p=0.1).to(device)
# encoder2= EncoderRNN(n_words, hidden_size).to(device)

# decoder2=DecoderRNN(hidden_size,n_words).to(device)
# # trainIters(encoder2, decoder2, 150,learning_rate=0.01)
# # torch.save(encoder2, "./encoder2.pt")
# # torch.save(decoder2, "./decoder2.pt")

# # trainIters(encoder1, attn_decoder1, 150,learning_rate=0.01)
# # torch.save(encoder1, "./encoder1.pt")
# # torch.save(attn_decoder1, "./attn_decoder1.pt")
def trainmodelIters(model,data_loader, n_iters, optimizer):
    # training_pairs = [pair for pair in dataloader]
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

    criterion = nn.NLLLoss()
    for i in range(n_iters):
        summary_loss = AverageMeter()
        data=tqdm(data_loader)
        for Train,Test in data:
            input_tensor = Train.T.to(device,non_blocking=True)
            target_tensor = Test.T.to(device,non_blocking=True)
            optimizer.zero_grad()
            outputs=model(input_tensor,target_tensor).permute(1,2,0)
            loss=criterion(outputs,target_tensor.T)
            #print([dataloader.tokenizer.decoder[i] for i in list(outputs[:,0].tolist())])
            loss.backward()
            optimizer.step()
            
            summary_loss.update(loss.item())
            data.set_postfix(AVGloss=summary_loss.avg,loss=loss.item())
        scheduler.step()
def evaluate(model,dataloader):
    with torch.no_grad():
            
        criterion = nn.NLLLoss()
        for i in range(n_iters):
            summary_loss = AverageMeter()
            data=tqdm(dataloader)
            for Train,Test in data:
                input_tensor = Train.T.to(device,non_blocking=True)
                target_tensor = Test.T.to(device,non_blocking=True)
                optimizer.zero_grad()
                outputs=model(input_tensor,target_tensor).permute(1,2,0)
                loss=criterion(outputs,target_tensor.T)
                #print([dataloader.tokenizer.decoder[i] for i in list(outputs[:,0].tolist())])
                #print(decode ans)
                summary_loss.update(loss.item())
                data.set_postfix(AVGloss=summary_loss.avg,loss=loss.item())
                yield outputs
#dataloader=myTrainQSPairs(context_length=20,dir=dir,QFile="./questions.txt",)
dataloader=myTrainASNQPairs(context_length=20,dir=dir,QFile="train.tsv")
#sampler=dataloader.getSampler(Batch)
###data_loader = torch.utils.data.DataLoader(dataloader,
#                                          num_workers=8,
#                                          batch_sampler=sampler,
#                                          prefetch_factor=3,
#                                          pin_memory=True)
data_loader=torch.utils.data.DataLoader(dataloader,
                                          batch_size=Batch, 
                                          num_workers=8,
                                          #shuffle=True,
                                          prefetch_factor=3,
                                          drop_last=False,
                                          pin_memory=True)                                         
n_words=len(dataloader.tokenizer.encoder)
SOT=dataloader.sot_token
EOT=dataloader.eot_token
ANS=dataloader.ans_token
REST=dataloader.rest_token
hidden_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=LSTM(hidden_size,n_words,SOT).cuda().to(device)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001,eps= 1e-3)
trainmodelIters(model,data_loader, 150,optimizer)
torch.save(model, "./QNLIModel.pt")

testdata_loader = torch.utils.data.DataLoader(dataloader,
                                          batch_size=Batch, 
                                          num_workers=8,
                                          #shuffle=True,
                                          prefetch_factor=3,
                                          drop_last=False,
                                          pin_memory=True)
# encoder2=torch.load( "./encoder2.pt")
# decoder2=torch.load("./decoder2.pt")
# #trainIters(encoder1, attn_decoder1, 150,learning_rate=0.01)
# encoder1=torch.load( "./encoder1.pt")
# attn_decoder1=torch.load("./attn_decoder1.pt")
for i in evaluate(model,testdata_loader):

     print("With Out: {}".format(dataloader.tokenizer.decode(eva[0])))
#     print("Without Attn: {}".format(dataloader.tokenizer.decode(eva2)))
