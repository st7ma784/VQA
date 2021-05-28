from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os,sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT,"CLIP")))

import CLIP.clip as CLIP
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size) #hiddensize.hidddensize

#     def forward(self, input, hidden):
#         #print(input.size())
#         output = self.embedding(input)
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.relu= nn.ReLU()
#     def forward(self, input, hidden,enc):
#         #print(input.shape) #b
#         output = self.embedding(input) #b,H
#         #print(output.shape) #b,H
#         output = self.relu(output)  #b,H
#         #print(output.shape)#b,H
#         output, hidden = self.gru(output.unsqueeze(1), hidden.unsqueeze(0))
#         #print(output.shape)#b,1,H
#         output=self.out(output.squeeze())
#         #print(output.shape)#b,[nwords]

#         output = self.softmax(output)
#         return output, hidden,[]

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=20,b=1):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size) #load from CLIP
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#         self.B=b
#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, input, hidden, encoder_outputs,Training=True):
#         #[B],[B,256],[B,20,256]
#         embedded = self.embedding(input)
#         B=input.size()[0]
#         #print("Embedded shape {}".format(embedded.shape)) #[B,256]
#         if Training:
#             embedded = self.dropout(embedded)
#         #print(embedded.shape) #[B,256]
#         #print(hidden.shape) #[B,256]

#         attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1).view(B,1,self.max_length)
#         #print(attn_weights.shape) #B,1,20
#         attn_applied = torch.bmm(attn_weights,encoder_outputs)
#         #print(attn_applied.shape)#b,1,256
#         output = torch.cat((embedded, attn_applied[:,0]), 1)
#         output = self.attn_combine(output)#.unsqueeze(1)
#         output = F.relu(output)
#         #print(output.shape) #[B,256]
#         #for i in range(output.size(0)):
#         output, hidden = self.gru(output.unsqueeze(0), hidden.unsqueeze(0))#.unsqueeze(0).unsqueeze(0))
#         #output[i,:,:] = F.log_softmax(self.out(output[i,:,:]), dim=1)
    
#         output =self.softmax(self.out(output[:])).squeeze() #this throws off B size 1? [:,0,:]
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

class LSTMEncoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size,output_size):
        super(LSTMEncoderRNN, self).__init__()
        self.embedding = embedding#nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size) #hiddensize.hidddensize

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        return output, hidden
class LSTMDecoderRNN(nn.Module):
    def __init__(self, embedding,hidden_size, output_size):
        super(LSTMDecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu= nn.ReLU()
        self.embedding=embedding#nn.Embedding(output_size, hidden_size)
    def forward(self, input, hidden,enc):
        output = self.embedding(input) #b,H
        output = self.relu(output)  #b,H
        output, hidden = self.gru(output.view(output.size(0),1,-1), hidden.view(1,hidden.size(0),-1))
        output=self.out(output.squeeze())
        output = self.softmax(output)
        return output, hidden,[]
class LSTMAttnDecoderRNN(nn.Module):
    def __init__(self,embedding, hidden_size, output_size, dropout_p=0.05, max_length=77):
        super(LSTMAttnDecoderRNN, self).__init__()
        self.max_length = max_length
        self.embedding =embedding ##nn.Embedding(output_size, hidden_size) #load from CLIP
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size * 2, hidden_size)

        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        #self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, input, hidden,positional_embedding, encoder_outputs,Training=True):
        embedded = self.embedding(input)
        hidden=F.softmax(self.attn2(torch.cat((positional_embedding, hidden), 1)), dim=1)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        attn_applied = attn_weights*encoder_outputs
        output = torch.cat((embedded, attn_applied.view(attn_applied.size(0),-1)), 1)        #[:,0
        output = self.attn_combine(output)#.unsqueeze(1)
        output = F.relu(output)
        output, hidden = self.gru(output.view(1,output.size(0),-1), hidden.view(1,hidden.size(0),-1))#self.gru(output.unsqueeze(0), hidden.unsqueeze(0))#.unsqueeze(0).unsqueeze(0))
        output=self.out(output)
        output =self.softmax(output[0]) #this throws off B size 1? [:,0,:]
        hidden=hidden.squeeze()
        return output, hidden, attn_weights #B,512,

    def forward(self, input, hidden, encoder_outputs,Training=True):
        #print(input.shape)
        embedded = self.embedding(input)
        #B=input.size(0)
        #embedded = self.dropout(embedded)
        #print("Emb {}".format(embedded.size()))#B,512
        #print("hidden {}".format(hidden.size()))#B,512

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        #print("attn_weights {}".format(attn_weights.size()))#200,512?

        #print("encoder_outputs {}".format(encoder_outputs.size()))#200,512

        attn_applied = attn_weights*encoder_outputs
        #print(attn_applied.shape)
        output = torch.cat((embedded, attn_applied.view(attn_applied.size(0),-1)), 1)        #[:,0

        output = self.attn_combine(output)#.unsqueeze(1)
        output = F.relu(output)
        output, hidden = self.gru(output.view(1,output.size(0),-1), hidden.view(1,hidden.size(0),-1))#self.gru(output.unsqueeze(0), hidden.unsqueeze(0))#.unsqueeze(0).unsqueeze(0))
        output=self.out(output)
        #print(output.size())
        output =self.softmax(output[0]) #this throws off B size 1? [:,0,:]
        hidden=hidden.squeeze()

        return output, hidden, attn_weights #B,512,  
class LSTMAttnDecoderRNNold(nn.Module):
    def __init__(self,embedding, hidden_size, output_size, dropout_p=0.05, max_length=20):
        super().__init__()
        self.max_length = max_length
        self.embedding =embedding ##nn.Embedding(output_size, hidden_size) #load from CLIP
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        #self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden, encoder_outputs,Training=True):
        #print(input.shape)
        embedded = self.embedding(input)
        B=input.size()[0]
        #embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1).view(B,1,self.max_length)
        attn_applied = torch.bmm(attn_weights,encoder_outputs)
        #print(attn_applied.shape)
        output = torch.cat((embedded, attn_applied.view(attn_applied.size(0),-1)), 1)        #[:,0

        output = self.attn_combine(output)#.unsqueeze(1)
        output = F.relu(output)
        output, hidden = self.gru(output.view(1,output.size(0),-1), hidden.view(1,hidden.size(0),-1))#self.gru(output.unsqueeze(0), hidden.unsqueeze(0))#.unsqueeze(0).unsqueeze(0))
        output =self.softmax(self.out(output[:])).squeeze() #this throws off B size 1? [:,0,:]
        return output, hidden, attn_weights

class LSTMGeneratorRNN(nn.Module):
    def __init__(self, embedding, hidden_size,output_size):
        super(LSTMGeneratorRNN, self).__init__()
        #embedding to Output...
        self.QGen=LSTMDecoderRNN(
                embedding=embedding,#nn.Embedding(output_size, hidden_size),
                hidden_size=hidden_size,
                output_size=output_size
            )
        self.AGen=LSTMDecoderRNN(
                embedding=embedding,#nn.Embedding(output_size, hidden_size),
                hidden_size=hidden_size,
                output_size=output_size
            )
        self.QEncoder=LSTMEncoderRNN(
                embedding=embedding,#nn.Embedding(output_size, hidden_size),
                hidden_size=hidden_size,
                output_size=output_size
        )

    def forward(Batch):
        #for batch of noise : convert noise to Q
        q=self.QGen(Batch)
        y=self.QEncoder(x)
        ans=self.AGen(y)
        return torch.cat(q,ans)
        #for given Q
        #gen A
        #return splices
class LSTMDiscriminatorRNN(nn.Module):
    def __init__(self, embedding,hidden_size,output_size):
        super(LSTMDiscriminatorRNN,self).__init__()
        self.encoder=LSTMEncoderRNN(
                embedding=embedding,#nn.Embedding(output_size, hidden_size),
                hidden_size=hidden_size,
                output_size=output_size
            )
        self.Lin = nn.Conv1d(512, 32)

        self.drop = nn.Dropout(0.2)
        self.act = nn.ReLU()
        self.out = nn.Linear(1)
    def forward(self,input):
        x = self.encoder(input) # [batch_size,hiddensize]
        x= self.act(self.Lin(x))
        x = self.drop(x)
        x = self.out(x) # [batch_size, 1]
        return x

class LSTMGAN(nn.Module):
    def __init__(self, embedding, hidden_size,output_size):
        super(LSTMGAN, self).__init__()
        self.Generator=LSTMGreneratorRNN(embedding, hidden_size,output_size)
        
        self.Discriminator=LSTMDiscriminatorRNN(embedding, hidden_size,output_size)
        self.hids=hidden_size
        self.optimizerD = optim.Adam(self.Discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.Generator.parameters(), lr=lr, betas=(beta1, 0.999))
    def forward(Batch, Questions=[],Ans=[],labels=[]):
        # Initialize BCELoss function
        b_size = Batch.size(0)
        criterion = nn.BCELoss()
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(b_size, self.hids, 1, 1, device=device)
        # Establish convention for real and fake labels during training
        real_label = 1. #                                                               labels
        fake_label = 0.
        #Defining the Adversarial ground truths 
        self.optimizerD.zero_grad()
        # Format batch
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output =  self.Discriminator(Batch).view(-1)                                    #Run on Q+A
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.hids, 1, 1, device=device)
        # Generate fake image batch with G
        fake = self.Generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = self.Discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.optimizerG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.Discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()


class LSTM(nn.Module):
    def __init__(self,
    hidden_size,
    output_size,
    SOT,
    dropout_p=0.1,
    max_length=20,
    teacher_forcing_ratio=0.5):   
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        Pretrained=CLIP.load("ViT-B/32", device=device, jit=False)[0].state_dict()["token_embedding.weight"]
        additionalsymbols=nn.Embedding(2,hidden_size).to(device)
        pretrained=torch.cat((Pretrained,additionalsymbols.weight))        
        print("{}".format(pretrained.size()))#[49408,512]
        dtype=torch.get_default_dtype()
        self.embedding.weight=nn.Parameter(pretrained.type(dtype))

        self.context_length = hidden_size
        self.encoder = LSTMEncoderRNN(
                embedding=self.embedding,#nn.Embedding(output_size, hidden_size),
                hidden_size=hidden_size,
                output_size=output_size

            )
        self.decoder=LSTMAttnDecoderRNN(
           embedding=self.embedding,
           hidden_size=hidden_size,
           output_size=output_size,
        )
        self.target_length=max_length
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.teacher_forcing_ratio=0.5
        self.SOT=SOT
    def forward(self, input_tensor, target_tensor):
        Batch= input_tensor.size(1)
        encoder_hidden =torch.zeros(1, Batch, self.hidden_size, device=device)
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)# [20,b,256],[1,b,256]
        encoder_outputs = torch.zeros(encoder_output.shape, device=device)#20,b,256
        encoder_outputs[:] = encoder_output[0,:,:]
        encoder_outputs=encoder_outputs.permute(1,0,2) #[b,20,256]
        decoder_input = torch.tensor([self.SOT]*Batch, device=device) #[B]
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        outputs=torch.zeros((target_tensor.size(0),target_tensor.size(1),self.output_size),device=device)
        if use_teacher_forcing:
            for di in range(self.target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden[0], encoder_outputs)
                outputs[di]=decoder_output
                decoder_input = target_tensor[di].detach() # Teacher forcing [B]
                #print("Pass Forces: {}".format(decoder_input[0]))
        else:
            for di in range(self.target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden[0], encoder_outputs)
                outputs[di]=decoder_output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                #print("Pass notForces: {}".format(decoder_input[0]))
        return outputs

class CLIPLSTM(nn.Module):
    def __init__(self,clip,dropout_p=0.1,max_length=77,teacher_forcing_ratio=0.5):   

        super().__init__()
        self.embedding = nn.Embedding(clip.embed_dim, clip.embed_dim)
        Pretrained=clip.state_dict()["token_embedding.weight"]
        #dditionalsymbols=nn.Embedding(2,model.embed_dim).to(device)
        #pretrained=torch.cat((Pretrained,additionalsymbols.weight))        
        #print("{}".format(pretrained.size()))#[49408,512]
        dtype=torch.get_default_dtype()
        self.embedding.weight=nn.Parameter(Pretrained.type(dtype),requires_grad=True)
        self.clip=clip#.eval()
        self.context_length = clip.context_length
        #self.imageEncoder = model.encode_image
        #self.textEncoder=model.encode_text
        self.device=device
        self.decoder=LSTMAttnDecoderRNN(
           embedding=self.embedding,
           hidden_size=clip.embed_dim,
           output_size=Pretrained.size(0),
        )
        self.target_length=35
        self.hidden_size=clip.embed_dim
        self.output_size=Pretrained.size(0)
        self.teacher_forcing_ratio=0.5
        self.padValue=0
    def forward(self, image_tensor, QTensor,GTQA_tensor): # 
        Batch= image_tensor.size(0)
        #print(image_tensor.size())#torch.Size([B, 3, 224, 224])
        #with torch.no_grad():
        encoder_output = self.clip.encode_image(image_tensor).float()
        # normalized features
        encoder_output = encoder_output / encoder_output.norm(dim=-1, keepdim=True)
        #encoder_output = encoder_output * self.model.logit_scale.exp()
        encoder_hidden = self.clip.encode_text(QTensor.T).float()
        encoder_hidden = encoder_hidden / encoder_hidden.norm(dim=-1, keepdim=True)
        #encoder_hidden = encoder_hidden *self.model.logit_scale.exp()
        #encoder_output,encoder_hidden=self.model(image_tensor,QTensor)#imageEncoder(image_tensor) #outputsize: n xhiddensize 
        #print(encoder_output.size())#B,512
        #print(encoder_hidden.size())#B,512
        
        #encoder_hidden =torch.zeros(1, Batch, self.hidden_size, device=device)
        #encoder_hidden=self.textEncoder(QTensor)
        #encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)# [20,b,256],[1,b,256]
        #encoder_outputs = torch.zeros(encoder_output.shape, device=device)#20,b,256
        #encoder_outputs[:] = encoder_output[0,:,:]
        #encoder_outputs=encoder_outputs.permute(1,0,2) #[b,20,256]
        #print(QTensor.size())#B,Contextlen
        decoder_input=QTensor[0] #symbols B
        #print(decoder_input.size())#B
        
        #decoder_input = torch.tensor([self.SOT]*Batch, device=device) #[B]
        decoder_hidden = encoder_hidden.float() #B,512
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        EoQMask=QTensor==self.padValue
        outputs=torch.zeros((self.target_length,Batch,self.output_size),device=self.device)
        #print(outputs.size())#77,B,512
        if use_teacher_forcing:
            for di in range(self.target_length):
                #print(decoder_hidden.size())#B,512
                #print(decoder_input.size())#B

                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                #print(decoder_output.size())#B,512
                #print(decoder_hidden.size())#B,512

                outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                decoder_input = GTQA_tensor[di].detach() # Teacher forcing [B]
                #print("dec : {}".format(decoder_input.size()))
                #print("Pass Forces: {}".format(decoder_input[0]))
        else:
            for di in range(self.target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                _,topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                decoder_input[torch.logical_not(EoQMask[di])]=QTensor[di][torch.logical_not(EoQMask[di])]
                
                #print("dec : {}".format(decoder_input.size()))
                #print("Pass notForces: {}".format(decoder_input[0]))
        return outputs
class swCLIPLSTM(CLIPLSTM):

        def forward(self, image_tensor, QTensor,GTQA_tensor): 
            Batch= image_tensor.size(0)
            encoder_output =self.clip.encode_text(QTensor.T).float()
            encoder_hidden =self.clip.encode_image(image_tensor).float()
            decoder_input=QTensor[0] #symbols B         
            decoder_hidden = encoder_hidden.float() #B,512
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            EoQMask=QTensor==self.padValue
            outputs=torch.zeros((GTQA_tensor.size(0),GTQA_tensor.size(1),self.output_size),device=device)
            if use_teacher_forcing:
                for di in range(self.target_length-1):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                    outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                    decoder_input = GTQA_tensor[di].detach() # Teacher forcing [B]
            else:
                for di in range(self.target_length-1):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                    outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                    _,topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
            return outputs

class psCLIPLSTM(CLIPLSTM):
    def __init__(self,clip,dropout_p=0.1,max_length=77,teacher_forcing_ratio=0.5,Batch=100):   

        super().__init__(clip,dropout_p=dropout_p,max_length=max_length,teacher_forcing_ratio=teacher_forcing_ratio)
        self.lstm=nn.lstm(input_size=Batch,hidden_size=self.hidden_size,num_layers=1,bidirectional=True)
        #self.Pos_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, image_tensor, QTensor,GTQA_tensor): 
        Batch= image_tensor.size(0)
        encoder_output = self.clip.encode_image(image_tensor).float()
        encoder_output = encoder_output / encoder_output.norm(dim=-1, keepdim=True)
        encoder_hidden = self.clip.encode_text(QTensor.T).float()
        encoder_hidden = encoder_hidden / encoder_hidden.norm(dim=-1, keepdim=True)
        #decoder_input=QTensor[0] #symbols B
        encoder_hidden=self.clip.positional_embedding +encoder_hidden
        encoder_output=self.clip.positional_embedding+encoder_output
        return self.lstm(QTensor[0],(encoder_hidden.float(),encoder_output))
        decoder_hidden = encoder_hidden.float() #B,512
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        EoQMask=QTensor==self.padValue
        outputs=torch.zeros((self.target_length,Batch,self.output_size),device=device)
        #print(outputs.size())#77,B,512
        if use_teacher_forcing:
            for di in range(self.target_length):
                #print(decoder_hidden.size())#B,512
                #print(decoder_input.size())#B

                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                #print(decoder_output.size())#B,512
                #print(decoder_hidden.size())#B,512

                outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                decoder_input = GTQA_tensor[di].detach() # Teacher forcing [B]
                #print("dec : {}".format(decoder_input.size()))
                #print("Pass Forces: {}".format(decoder_input[0]))
        else:
            for di in range(self.target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output)
                outputs[di][EoQMask[di]]=decoder_output[EoQMask[di]]
                _,topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                decoder_input[torch.logical_not(EoQMask[di])]=QTensor[di][torch.logical_not(EoQMask[di])]
                
                #print("dec : {}".format(decoder_input.size()))
                #print("Pass notForces: {}".format(decoder_input[0]))
        return outputs
