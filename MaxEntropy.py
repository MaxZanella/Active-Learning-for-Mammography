# -*- coding: utf-8 -*-
"""

Author: Maxime Zanella
"""
def query_uncertainty(model,unlabeled_loader,budget,device,measure="entropy"):
    model.eval()
    entropy_measure= np.zeros((len(unlabeled_loader)*model.params['batch_size'],))
    with torch.no_grad():
        i=0     
        for batch_idx, (data, target) in enumerate(unlabeled_loader):
            prob = np.zeros((data.shape[0],model.params['num_classes']))
            data,target = data.to(device),target.to(device)
            out = model(data)
            out = F.softmax(out,dim=1)
            out = out.cpu().numpy()
            entropy_measure[i:i+data.shape[0]] = entropy(out).squeeze()
            i = i + data.shape[0]
    entropy_measure = entropy_measure[:i]
    return np.argsort(entropy_measure)
