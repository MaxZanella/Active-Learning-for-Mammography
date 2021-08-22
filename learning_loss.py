# -*- coding: utf-8 -*-
"""
Author: Maxime Zanella
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def Learning_Loss(model,unlabeled_loader,budget,device):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model['main'].eval()
    model['module'].eval()
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for data, _ in unlabeled_loader:
            data = data.to(device)

            model['main'].layer1.register_forward_hook(get_activation("features1"))
            model['main'].layer2.register_forward_hook(get_activation("features2"))
            model['main'].layer3.register_forward_hook(get_activation("features3"))
            model['main'].layer4.register_forward_hook(get_activation("features4"))
            
            _ = model['main'](data)
            
            f1 = activation['features1']
            f2 = activation['features2']
            f3 = activation['features3']
            f4 = activation['features4']
            features = [f1, f2, f3, f4]
            
            pred_loss = model['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return np.argsort(uncertainty.cpu().numpy())
