# Last update: 26 November 2018
# 
# Author: Shaurya Shubham
# This code train the link prediction model by using HoLE and evaluate it
# 
# Inputs:
#     edges.csv : Dataset to create negative sample
#     train.csv : Dataset to train the model
# Outputs:
#     Statistics:
#        * About trainig
#        * About validation
#        * About the use of the model

####################################################################################################

# Loading Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from numpy.fft import fft, ifft
import random
import itertools
import pandas as pd
import tqdm

true_data = pd.read_csv('./Data/edges.csv')
Total_Entities = 4000
Total_Relations = 4
Batch_Size = 2

# Function to create positive and negative samples
def positive(fact):

    #print fact
    entity_a, relation, entity_b = fact['entity_a'], fact['relation'], fact['entity_b']
    entity_a_id = Variable(torch.LongTensor([entity_a]))#.view(1, -1)
    entity_b_id = Variable(torch.LongTensor([entity_b]))#.view(1, -1)
    relation_id = Variable(torch.LongTensor([relation]))#.view(1, -1)
    target = Variable(torch.LongTensor([1])).view(1, -1)
    return [entity_a_id,entity_b_id,relation_id,target]
    
def negative():
    # Sample until we find an invalid fact
    while True:
        entity_a = random.randint(0, Total_Entities - 1)
        relation = random.randint(0, Total_Relations - 1)
        entity_b = random.randint(0, Total_Entities - 1)
        target = Variable(torch.LongTensor([0])).view(1, -1)
        rule = {'entity_a':[entity_a],'entity_b':[entity_b],'relation':[relation]}
        if len(list(true_data.index[true_data.isin(rule).all(1)].values)) == 0:
            break

    entity_a_id = Variable(torch.LongTensor([entity_a]))#.view(1, -1)
    relation_id = Variable(torch.LongTensor([relation]))#.view(1, -1)
    entity_b_id = Variable(torch.LongTensor([entity_b]))#.view(1, -1)
    
    return [entity_a_id,entity_b_id,relation_id,target]

# Custom Dataloader
class CustomDataset():
    
    def __init__(self,filepath):
        self.data = pd.read_csv(filepath, delimiter=',')
        self.data.dropna(inplace=True)
        
    def __getitem__(self,index):
        sample = {}
        sample['pos'] = positive(self.data.iloc[index])
        sample['neg'] = negative()
        return [sample['pos'],sample['neg']]
        
    def __len__(self):
        return (self.data.shape[0])

hole_data = CustomDataset('./Data/train.csv')

# Collage function to handle lists in dataloader
def my_collate(batch):
    batch = list(itertools.chain.from_iterable(batch))
    data = [item for item in batch]
    return data

dataloader = DataLoader(hole_data, batch_size=Batch_Size,shuffle=False, num_workers=4,collate_fn=my_collate)

## Holographic Embedding Implemetations

class HoLE(nn.Module):
    
    def __init__(self,num_entity,num_rel, emb_dim):
        super(HoLE,self).__init__()
        self.ent_embeddings=nn.Embedding(num_entity,emb_dim)
        self.rel_embeddings=nn.Embedding(num_rel,emb_dim)
        #self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
    
    # Circular correlation
    def ccorr(self, a, b):
        #print a
        #print a.size()
        k = torch.rfft(a,1,onesided=False)
        t = torch.rfft(b,1,onesided=False)
        k = k *torch.FloatTensor([1,-1])
        real = (k[:,:,0] * t[:,:,0]) - (k[:,:,1] * t[:,:,1])
        imag = (k[:,:,0] * t[:,:,1]) + (k[:,:,1] * t[:,:,0])
        a=[]
        for i,j in zip(real,imag):
            a.append(i)
            a.append(j)
        a =torch.stack(a,dim=-1)
        #print a.size()
        t = torch.split(a,2,dim=1)
        t = torch.stack(t,dim=0)
        #print t.size()
        v = torch.ifft(t,1)
        #print v.size()
        return v[:,:,0]
    
    def score(self,head, tail, rel):
        entity_mention = self.ccorr(head, tail)
        relation_norm = rel.norm(p=2, dim=1, keepdim=True)
        relation_mention = rel.div(relation_norm.expand_as(rel))
        #print relation_mention.size()
        #print entity_mention.size()
        _sum = torch.sum(relation_mention * entity_mention,dim=1)
        #print _sum.size()
        return torch.sigmoid(_sum)
    
    def forward(self,x):
        #x = Variable(torch.LongTensor(x))
        #print x
        #print type(x)
        entity_a, entity_b, relation = x[:,0].view(-1, 1),x[:,1].view(-1, 1),x[:,2].view(-1, 1)
        s = self.ent_embeddings(entity_a)
        #print entity_b
        o = self.ent_embeddings(entity_b)
        r = self.rel_embeddings(relation)
        #print s.view(-1,10).size()
        #print o.size()
        #print r.size()
        out = self.score(s.view(-1,emb_dim),o.view(-1,emb_dim),r.view(-1,emb_dim))
        return out

## Loading the model
num_entity = Total_Entities
num_relation = Total_Relations
emb_dim =5
hole = HoLE(num_entity,num_relation,emb_dim)
print hole

# Total number of trainable parameters
total_params = sum(p.numel() for p in hole.parameters() if p.requires_grad)
print("Total number of trainable parameters : {}" .format(total_params))

# Criterion and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(hole.parameters(), lr=0.001)

# train and validation losses
train_losses = []
valid_losses = []

# Training of model
num_epoch = 350

for epoch in range( num_epoch):
    
    print('Epoch {}/{}'.format(epoch, num_epoch- 1))
    print('-' * 10)
    
    # Each epoch has a training 
    hole.train() # Set model to training mode
    
    running_loss = 0.0
    # Iterate over data.
    for idx,x in enumerate(dataloader):
        x = Variable(torch.LongTensor(x))
        #print x.size()
        outputs = hole(x) # Forward pass: compute the output class given a image
        #print outputs.size()
        #break
        target = x[:,3].view(-1, 1)
        target = target.type(torch.FloatTensor)
        optimizer.zero_grad() # clear gradients for next train
        loss = criterion(outputs,target) # Compute the loss: difference between the output class and the pre-given label
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients  and update the weights of hidden nodes

        running_loss += loss.data / float(Batch_Size*2)

        #if phase == 'train':
        #    if (i+1) % 100 == 0 :
        #        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, i+1, data_lengths[phase]//64, loss.data))

    epoch_loss = running_loss 
    train_losses.append(epoch_loss)
    print('Epoch [{}/{}]{} Loss: {:.4f}'.format(epoch+1, num_epoch,'train', epoch_loss))
    

# Plot training loss

import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(8, 6),)
plt.plot(train_losses)
plt.xlabel('epochs')
plt.ylabel('loss');

# prediction function
def hole_prediction(model,fact):
    model.eval()
    i = fact['entity_a']
    j = fact['entity_b']
    score = -1
    for k in range(4):
        xt = [torch.LongTensor([i]),torch.LongTensor([j]),torch.LongTensor([k])]
        xt = Variable(torch.LongTensor(xt))
        xt = xt.view(1,-1)
        #print xt.size()
        #print "predicting"
        pred_score = hole(xt)
        #print pred_score
        if score < pred_score.data:
            prediction = k
            score = pred_score.data
    return score, prediction

## Training Data Stats

train_data =  pd.read_csv('./Data/train.csv', delimiter=',')
train_data.dropna(inplace=True)
train_data['prediction'] = ''
# Calculating train data stats
for idx,dat in  tqdm.tqdm(enumerate(train_data.iterrows())):
    _,train_data['prediction'].iloc[idx] = hole_prediction(hole,dat[1])

acc_count = (train_data['prediction'] == train_data['relation']).sum()
acc = float(acc_count)/train_data.shape[0]

# Printing results
print("\n- - - - - - - - - - - - - STATISTICS ON TRAINING DATASET - - - - - - - - - - - - - - \n")
print("Total number of Green links in training dataset : {} ".format(train_data.shape[0]))
print(" Number of Green predicted correctly: {}".format(acc_count))
print(" Accuracy of Green predicted correctly: {}".format(acc))

## Validation Data Stats

valid_data =  pd.read_csv('./Data/valid.csv', delimiter=',')
valid_data.dropna(inplace=True)
valid_data['prediction'] = ''
# Calculating train data stats
for idx,dat in  tqdm.tqdm(enumerate(valid_data.iterrows())):
    _,valid_data['prediction'].iloc[idx] = hole_prediction(hole,dat[1])

acc_count = (valid_data['prediction'] == valid_data['relation']).sum()
acc = float(acc_count)/valid_data.shape[0]

# Printing results
print("\n- - - - - - - - - - - - - STATISTICS ON VALIDATION DATASET - - - - - - - - - - - - - - \n")
print("Total number of Green links in training dataset : {} ".format(valid_data.shape[0]))
print(" Number of Green predicted correctly: {}".format(acc_count))
print(" Accuracy of Green predicted correctly: {}".format(acc))

## To_use Data Stats

test_data =  pd.read_csv('./Data/to_use.csv', delimiter=',')
test_data.dropna(inplace=True)
test_data['prediction'] = ''
# Calculating train data stats
for idx,dat in  tqdm.tqdm(enumerate(test_data.iterrows())):
    _,test_data['prediction'].iloc[idx] = hole_prediction(hole,dat[1])

acc_count = (test_data['prediction'] == test_data['relation']).sum()
acc = float(acc_count)/test_data.shape[0]

# Printing results
print("\n- - - - - - - - - - - - - STATISTICS ON TO_USE DATASET - - - - - - - - - - - - - - \n")
print("Total number of Green links in training dataset : {} ".format(test_data.shape[0]))
print(" Number of Green predicted correctly: {}".format(acc_count))
print(" Accuracy of Green predicted correctly: {}".format(acc))
