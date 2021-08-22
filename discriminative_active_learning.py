# -*- coding: utf-8 -*-
"""
Author: Maxime Zanella
Inspired from https://github.com/dsgissin/DiscriminativeActiveLearning
"""
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os

class DiscriminativeModel(nn.Module):
    def __init__(self,params,input_size):
        super(DiscriminativeModel,self).__init__()
        self.input_size = input_size
        self.params = params
        if input_size < 30:
            width = 20
            self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,2),
            )
        else:
            width=256
            self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,2),
            )
    def forward(self,x):
        out = self.model(x)
        return out

class DiscriminativeDataset(torch.utils.data.Dataset):
    def __init__(self,length,split,data_folder="discriminative_features"):
        self.transform = transforms.ToTensor()
        self.length = length
        self.data_folder = data_folder
        self.n_train = split[0]
        self.n_unlabeled = split[1]
    def __len__(self):
        return self.length
    def __getitem__(self,index):   
        features = np.load(os.path.join(self.data_folder,"features"+str(index)+'.npy'))
        target = features[features.shape[0]-1]
        data = np.expand_dims(features[:features.shape[0]-1],axis=0)
        #print(data.shape)
        data = self.transform(data).float()
        #print(data.size(),target)
        return data,int(target)
    
def create_dataset_discriminative(features_train, features_unlabeled,data_folder="discriminative_features"):
    len_features = features_train.shape[1]
    n_train = 0
    n_unlabeled = 0
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for i in range(features_train.shape[0]):
        features = np.zeros((len_features+1,))
        features[:len_features] = features_train[i,:]
        features[len_features] = 0
        n_train += 1
        np.save(os.path.join(data_folder,"features"+str(i)+".npy"),features)
    for index,i in enumerate(range(features_train.shape[0],features_train.shape[0]+features_unlabeled.shape[0])):
        features = np.zeros((len_features+1,))
        features[:len_features] = features_unlabeled[index,:]
        features[len_features] = 1
        n_unlabeled += 1
        np.save(os.path.join(data_folder,"features"+str(i)+".npy"),features)
    dataset = DiscriminativeDataset(length=features_train.shape[0]+features_unlabeled.shape[0],split=(n_train,n_unlabeled))
    return dataset
def train_discriminative(model,dataset,device):
    THRESHOLD = 0.98
    batch_size = 1024
    if model.input_size == 28:
        optimizer = optim.Adam(model.parameters(),lr=0.0003)
        epochs = 200
    elif model.input_size == 128:
        batch_size = 128
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        epochs = 200
    elif model.input_size == 512:
        batch_size = 128
        optimizer = optim.Adam(model.parameters(),lr=0.0002,weight_decay=5e-4)
        epochs = 500
    else:
        optimizer = optim.Adam(model.parameters(),lr=0.0003)
        epochs = 1000
        batch_size = 32
        
    weights = [dataset.n_unlabeled,dataset.n_train]
    weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(dataset, 
    batch_size = batch_size,
    shuffle=True, 
    pin_memory=model.params['pin_memory'], 
    num_workers = model.params['num_workers'],
    )
    model.train()
    tot_loss = 0.0
    accuracy = 0.0
    length = 0
    best_accuracy = 0
    no_improvement = 0
    for epoch in range(epochs):
        accuracy = 0.0
        length = 0
        for batch_idx,(x,y) in enumerate(loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            length += x.size(0)
            out = model(x)
            loss = loss_fn(out,y)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()
            out = F.softmax(out,dim=1)


            _,pred = torch.max(out,1)
            accuracy += torch.sum(pred == y).item()
        no_improvement += 1
        accuracy = accuracy/length
        if accuracy>best_accuracy:
            no_improvement = 0
            best_accuracy = accuracy
            torch.save(model,'best_discriminative.pth')
        
        if accuracy>THRESHOLD and epoch>2:
            torch.save(model,'best_discriminative.pth')
            return
        if no_improvement>30:
            return
        
    
    
        
global view_output
def hook_fn(module, input, output):
    global view_output
    view_output = output    
def query_discriminative(model,train_loader_no_transform, unlabeled_loader, input_shape, device):
    model.layer4.register_forward_hook(hook_fn)
    model.eval()
    batch_size = model.params['batch_size']
    len_train = max(1,len(train_loader_no_transform))*batch_size
    len_data = len(unlabeled_loader)*batch_size
    len_features = 512
    features_train = np.zeros((len_train,len_features))
    features_unlabeled = np.zeros((len_data,len_features))
    len_train = 0
    len_unlabeled = 0
    i=0
    with torch.no_grad():
        for batch_idx,(x,_) in enumerate(train_loader_no_transform):
            x = x.to(device) 
            len_train += x.size(0)
            _ = model(x)
            features = F.avg_pool2d(view_output,4)
            features = features.view(features.size(0), -1)
    
            features = features.cpu().numpy()
            features_train[i:i+features.shape[0],:]=features
            i = i + features.shape[0]
        i=0 
        for batch_idx,(x,_) in enumerate(unlabeled_loader):
            x = x.to(device) 
            len_unlabeled += x.size(0)
            _ = model(x)
            features = F.avg_pool2d(view_output,4)
            features = features.view(features.size(0), -1)
    
            features = features.cpu().numpy()
            features_unlabeled[i:i+features.shape[0],:]=features
            i = i + features.shape[0]
    index_unlabeled = [i for i in range(len_train,len_train+len_unlabeled)]
    
    features_train = features_train[:len_train,:]
    features_unlabeled = features_unlabeled[:len_unlabeled,:]
    
    dataset = create_dataset_discriminative(features_train,features_unlabeled)
    
    discriminative_model = DiscriminativeModel(params = model.params,input_size=len_features).to(device)
    
    train_discriminative(discriminative_model,dataset,device)
    
    dataset_unlabeled = Subset(dataset,indices=index_unlabeled)
    
    loader = DataLoader(
            dataset_unlabeled,
            batch_size = 1024,
            num_workers = model.params['num_workers'],
            pin_memory = model.params['pin_memory']
            )
    discriminative_model.eval()
    discriminative_model = torch.load('best_discriminative.pth')
    with torch.no_grad():
        for batch_idx,(x,_) in enumerate(loader):
            x = x.to(device)
            out = discriminative_model(x)
            out = out.cpu().detach().numpy()
            if batch_idx == 0:
                output_concat = out
            else:
                output_concat = np.concatenate((output_concat,out))
    
    arg = np.argsort(output_concat[:,1])
    del discriminative_model
    return arg

    

if __name__=="__main__":
    model = DiscriminativeModel(input_shape=(1,28,28))
    x = torch.rand(64,1,28,28)
    output = model(x)
    print(output.size())


    
    

        