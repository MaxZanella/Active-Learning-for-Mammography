# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:01:42 2021

Author: Maxime Zanella
Inspired from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
"""
import numpy as np
from model import ClassicResNet
from learning_loss_net import Learning_Loss_Net, LossPredLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 

    
def train_epoch(train_loader, model, optimizer, epoch, device, scheduler, print_bool):
    WEIGHT = 1.0
    MARGIN = 1.0
    loss = model['main'].params['loss']
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    model['main'].train()
    model['module'].train()
    
    model['main'].layer1.register_forward_hook(get_activation("features1"))
    model['main'].layer2.register_forward_hook(get_activation("features2"))
    model['main'].layer3.register_forward_hook(get_activation("features3"))
    model['main'].layer4.register_forward_hook(get_activation("features4"))

    for batch_idx, (data,target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer['main'].zero_grad()
        optimizer['module'].zero_grad()

        scores = model['main'](data)
        target_loss = loss(scores, target)
        
        f1 = activation['features1']
        f2 = activation['features2']
        f3 = activation['features3']
        f4 = activation['features4']
        features = [f1, f2, f3, f4]

        if epoch > model['main'].params['epochs_loss']:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.

            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = model['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        final_loss            = m_backbone_loss + WEIGHT * m_module_loss

        final_loss.backward()
        optimizer['main'].step()
        optimizer['module'].step()
    if scheduler != None:
        scheduler['main'].step()
        scheduler['module'].step()
    
def test(model, loader, device):
    model['main'].eval()
    model['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            scores  = model['main'](data)
            scores = F.softmax(scores, dim=1)
            _, preds = torch.max(scores, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
    
    return 100 * correct / total


    
def search_active_resnet_learning_loss(params, loaders, n_models, device, print_bool=False):
    train_loader, val_loader = loaders
    best_model = None
    best_accuracy = 0
    for i in range(n_models):
        EPOCHS = 60
        LEARNING_RATE_INIT = 10**(-1 - 1*np.random.uniform()) # between 1e-4 and 1e-6
        WEIGHT_DECAY = 5e-4 #10**(-5 + 3*np.random.uniform()) # between 0.0001 and 1
        DROPOUT =  0.3 + 0.4*np.random.uniform()#params['dropout'] #0.3 + 0.5*np.random.uniform() # between 0.3 and 0.8
        LEARNING_RATE_DECAY = params['lr_decay']
        LAST_FROZEN_LAYER = params['last_frozen_layer']
        milestones = [33,45]
        N_STEPS = int(EPOCHS/3)
        EPOCHS_LOSS = 25
        #print(EPOCHS_LOSS)
        new_params = {
            'num_classes':params['num_classes'],
            'epochs':EPOCHS,
            'batch_size':params['batch_size'],
            'lr':LEARNING_RATE_INIT,
            'lr_decay':LEARNING_RATE_DECAY,
            'n_steps':N_STEPS,
            'weight_decay':WEIGHT_DECAY,
            'epochs_loss':EPOCHS_LOSS,
            'dropout':DROPOUT,
            'img_size':params['img_size'],
            'backbone_name':params['backbone_name'],
            'last_frozen_layer':LAST_FROZEN_LAYER,
            'loss': nn.CrossEntropyLoss(reduction='none'),
            'in_channels':params['in_channels'],
            'dest_dir':params['dest_dir'],
            'num_workers':params['num_workers'],
            'pin_memory':params['pin_memory']
            }
        if params['network']=='FineTunedResnet':
            backbone = FineTunedResnet(new_params).to(device)
        elif params['network']=='FullyConnected':
            backbone = FullyConnected(new_params).to(device)
        elif params['network']=='ClassicResNet':
            backbone = ClassicResNet(new_params).to(device)
        module = Learning_Loss_Net().to(device)
        model = {'main':backbone,'module':module}
        optimizer_backbone = optim.SGD(model['main'].parameters(), lr=new_params['lr'], 
                                    momentum=0.9, weight_decay=new_params['weight_decay'])
        optimizer_module   = optim.SGD(model['module'].parameters(), lr=new_params['lr'], 
                                    momentum=0.9, weight_decay=new_params['weight_decay'])
        #optimizer_backbone = optim.Adam(backbone.parameters(), lr=new_params['lr'],weight_decay=new_params['weight_decay'])
        #optimizer_module = optim.Adam(module.parameters(), lr=new_params['lr'],weight_decay=new_params['weight_decay'])
        optimizer = {'main':optimizer_backbone,'module':optimizer_module}
        if new_params['lr_decay']!=0:
            scheduler_backbone = optim.lr_scheduler.MultiStepLR(optimizer['main'], milestones = milestones, gamma=new_params['lr_decay'])
            scheduler_module = optim.lr_scheduler.MultiStepLR(optimizer['module'], milestones = milestones, gamma=new_params['lr_decay'])
            scheduler = {'main':scheduler_backbone,'module':scheduler_module}
        else: 
            scheduler = None
        
        for epoch in range(new_params['epochs']):
            train_epoch(train_loader, model, optimizer, epoch, device, scheduler, print_bool)
            accuracy_val = test(model, val_loader, device)
            #print(accuracy_val,'at epoch',epoch)
            if accuracy_val>best_accuracy:
                best_accuracy = accuracy_val
                torch.save({'backbone_state_dict':model['main'].state_dict()},'best_model_main.pth')
                torch.save({'module_state_dict':model['module'].state_dict()},'best_model_module.pth')
                if print_bool:
                    print("best accuracy :", accuracy_val)
        del model['main']
        del model['module']
            

    if params['network']=='FineTunedResnet':
         backbone = FineTunedResnet(new_params).to(device)
    elif params['network']=='FullyConnected':
        backbone = FullyConnected(new_params).to(device)
    elif params['network']=='ClassicResNet':
        backbone = ClassicResNet(new_params).to(device)
    module = Learning_Loss_Net().to(device)

    checkpoint_backbone = torch.load('best_model_main.pth')
    checkpoint_module = torch.load('best_model_module.pth')
    backbone.load_state_dict(checkpoint_backbone['backbone_state_dict'])
    module.load_state_dict(checkpoint_module['module_state_dict'])

    best_model = {'main':backbone,'module':module}

    return best_model
