# -*- coding: utf-8 -*-
"""

@author: maxim
"""
def entropy(labels):
    return np.sum(-(labels * np.log(labels)),axis=1,keepdims=True)

def MC_dropout(model,unlabeled_loader,budget,nb_MC_samples,device,measure="entropy"):
  model.train() #activate dropout
  if measure == 'var_ratio':
      return MC_dropout_var_ratio(model,unlabeled_loader,budget,nb_MC_samples,device)
  elif measure =='entropy':
      entropy_measure= np.zeros((len(unlabeled_loader)*model.params['batch_size'],))
      with torch.no_grad():
        i=0     
        for batch_idx, (data, target) in enumerate(unlabeled_loader):
            prob = np.zeros((data.shape[0],model.params['num_classes']))
            data,target = data.to(device),target.to(device)
            for _ in range(nb_MC_samples):
                out = model(data)
                out = F.softmax(out,dim=1)
                prob +=out.cpu().numpy()
            prob = prob/nb_MC_samples
            entropy_measure[i:i+data.shape[0]] = entropy(prob).squeeze()
            i = i + data.shape[0]
      entropy_measure = entropy_measure[:i]
      return np.argsort(entropy_measure)