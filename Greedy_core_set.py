# -*- coding: utf-8 -*-
"""

@author: maxim
"""
def GreedyKCenter(model,unlabeled_loader,train_loader_no_transform,k,device):
  G = []
  batch_size = model.params['batch_size']
  len_train = len(train_loader_no_transform)*batch_size
  len_data = len(unlabeled_loader)*batch_size
  len_features = 512
  feature_matrix = np.zeros((len_data,len_features))
  s = np.zeros((len_train,len_features))
  i=0    
  model.eval()
  length = 0
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(unlabeled_loader):
      length += data.size(0)
      data,target = data.to(device),target.to(device)

      model.layer4.register_forward_hook(hook_fn)
      _ = model(data)

      features = F.avg_pool2d(view_output,4)
      features = features.view(features.size(0),-1)
      features = features.cpu().numpy()
      feature_matrix[i:i+features.shape[0],:]=features
      i = i + features.shape[0]
    i=0
    length_train = 0
    for batch_idx, (data, target) in enumerate(train_loader_no_transform):
      length_train += data.size(0)
      data,target = data.to(device),target.to(device)
      _ = model(data)
      features = F.avg_pool2d(view_output,4)
      features = features.view(features.size(0),-1)
      features = features.cpu().numpy()
      s[i:i+features.shape[0],:]=features
      i = i + features.shape[0]
  len_data = length
  d = np.ones((len_data,1))*np.infty
  feature_matrix = feature_matrix[:len_data,:]
  s = s[:length_train,:]
  for l in range(len_data):
    for j in range(s.shape[0]):
      d[l] = min(d[l],np.linalg.norm(s[j,:]-feature_matrix[l,:]))
  u = np.argmax(d)
  G.append(u)
  for i in range(1,k):
    for l in range(len_data):
      d[l] = min(d[l],np.linalg.norm(feature_matrix[u,:]-feature_matrix[l,:]))
    u = np.argmax(d)
    G.append(u)
  G.sort()
  arg = []
  i = 0
  for elem in G:
    while i!= elem:
      arg.append(i)
      i=i+1
    i=i+1
  while i< length:
    arg.append(i)
    i=i+1
  for elem in G:
    arg.append(elem)

  return arg

def new_indices(indices,chosen_indices):
  chosen_indices.sort()
  new = []
  i = 0
  for elem in chosen_indices:
    while indices[i]!= elem:
      new.append(indices[i])
      i=i+1
    i=i+1
  while i<len(indices):
    new.append(indices[i])
    i=i+1
  return new