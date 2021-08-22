# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:12:39 2021

@author: maxim
"""
def query_ensembles_var(models,unlabeled_loader,budget,device):
    var_ratio_measure = np.zeros((len(unlabeled_loader)*models[0].params['batch_size'],))
    i=0 
    with torch.no_grad():    
        for batch_idx, (data, target) in enumerate(unlabeled_loader):
            data,target = data.to(device),target.to(device)
            n_labels_pred = np.zeros((data.size(0),models[0].params['num_classes']))
            for model in models:
                out = model(data)
                out = F.softmax(out,dim=1)
                out = out.cpu().numpy()
                pred_label = np.argmax(out, 1)
                for j in range(pred_label.shape[0]):
                    n_labels_pred[j,pred_label[j]] += 1
            var_ratio_measure[i:i+data.size(0)] = 1-np.max(n_labels_pred,1)/len(models)
        
            i = i + data.shape[0]
        
        
    var_ratio_measure = var_ratio_measure[:i]
    return np.argsort(var_ratio_measure)

def entropy(labels):
    return np.sum(-(labels * np.log(labels)),axis=1,keepdims=True)

def query_ensembles(models,unlabeled_loader,budget,device,measure="entropy"):
    for i in range(len(models)):
        models[i].eval()
    entropy_measure= np.zeros((len(unlabeled_loader)*models[0].params['batch_size'],))
    if measure=='var_ratio':
        return  query_ensembles_var(models,unlabeled_loader,budget,device)
    else:
        with torch.no_grad():
            i=0
            for batch_idx, (data, target) in enumerate(unlabeled_loader):
                prob = np.zeros((data.shape[0],models[0].params['num_classes']))
                data,target = data.to(device),target.to(device)
                for k in range(len(models)):
                    out = models[k](data)
                    out = F.softmax(out,dim=1)
                    prob +=out.cpu().numpy()
                prob = prob/len(models)
                entropy_measure[i:i+data.shape[0]] = entropy(prob).squeeze()
                i = i + data.shape[0]
        entropy_measure = entropy_measure[:i]
        for i in range(len(models)):
            models[i].train()
        return np.argsort(entropy_measure)