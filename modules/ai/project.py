import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import models

from torch.utils.data import DataLoader
from torchvision.transforms import v2

import os
from glob import glob
from tqdm import tqdm

train_df = pd.DataFrame({"path":[] , "label":[] , "class_id":[]})
train_path = 'data\\Training'
label_list = ['notsmoking','smoking']
img_list = glob(train_path+'\*.jpg')

for img in img_list:
    file_name = os.path.splitext(img)[0].split("\\")[-1]
    if file_name[:len(label_list[0])] == label_list[0]:
        new_data = pd.DataFrame({"path":img , "label":label_list[0] , "class_id":0} , index=[1])
        train_df = pd.concat([train_df , new_data] , ignore_index = True)
    elif file_name[0:len(label_list[1])] == label_list[1]:
        new_data = pd.DataFrame({"path":img , "label":label_list[1] , "class_id":1} , index=[1])
        train_df = pd.concat([train_df , new_data] , ignore_index = True)

train_df[["path"]] = train_df[["path"]].astype(str)
train_df[["label"]] = train_df[["label"]].astype(str)
train_df[["class_id"]] = train_df[["class_id"]].astype(int)

val_df = pd.DataFrame({"path":[] , "label":[] , "class_id":[]})
val_path = "data\\Validation"
img_list = glob(val_path + "\*.jpg")

for img in img_list:
    file_name = os.path.splitext(img)[0].split("\\")[-1]
    if file_name[:len(label_list[0])] == label_list[0]:
        new_data = pd.DataFrame({"path":img , "label":label_list[0] , "class_id":0} , index = [1])
        val_df = pd.concat([val_df , new_data]  , ignore_index=True)
    elif file_name[:len(label_list[1])] == label_list[1]:
        new_data = pd.DataFrame({"path":img , "label":label_list[1] , "class_id":1} , index = [1])
        val_df = pd.concat([val_df , new_data] , ignore_index=True)

val_df[['path']] = val_df[['path']].astype(str)
val_df[['label']] = val_df[['label']].astype(str)
val_df[['class_id']] = val_df[['class_id']].astype(int)

train_transforms = v2.Compose([
    v2.Resize(265),
    v2.RandomResizedCrop(size = (224 , 224) , antialias = True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(.5),
    v2.RandomAffine(degrees=(-10,10),translate=(.1,.1), scale=(.9,1.1)),
    v2.RandomErasing(p=.5,scale = (.1,.15)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean = [.485,.456,.406] , std = [0.229 , 0.224 , 0.225])
])

test_transforms = v2.Compose([
    v2.Resize((224,224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean = [.485,.456,.406] , std = [0.229 , 0.224 , 0.225])
])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self , dataframe , transforms_):
        self.df = dataframe
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self ,index):
        img_path = self.df.iloc[index]['path']
        img = Image.open(img_path).convert("RGB")
        transformed_img = self.transforms_(img)
        class_id = self.df.iloc[index]['class_id']
        return transformed_img , class_id

BATCH_SIZE = 6
device = torch.device('cpu')
#num_workers = 2 if device == 'cuda' else 4

train_dataset = MyDataset(train_df , train_transforms)
val_dataset = MyDataset(val_df , test_transforms)
train_loader = DataLoader(train_dataset , batch_size=BATCH_SIZE , shuffle = True)
val_loader = DataLoader(val_dataset , batch_size=BATCH_SIZE)

class_size = len(label_list)
model = models.swin_v2_b(weights= 'DEFAULT')

model.head = nn.Linear(in_features = model.head.in_features,
                       out_features = class_size)

def train(dataloader , model , loss_fn , optimizer , lr_scheduler):
    size = 0
    num_batches = len(dataloader)
    
    model.train()
    epoch_loss , epoch_correct = 0 , 0
    
    for i ,(data_ , target_) in enumerate(dataloader):
        target_ = target_.type(torch.LongTensor)
        data_ , target_ = data_.to(device) , target_.to(device)
        
        outputs = model(data_)
        
        loss = loss_fn(outputs , target_)
        epoch_loss =+ loss.item()
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _ , pred = torch.max(outputs , dim = 1)
        epoch_correct = epoch_correct + torch.sum(pred == target_).item()
        size += target_.shape[0]
    lr_scheduler.step()
    return epoch_correct/size , epoch_loss / num_batches

def test(dataloader , model , loss_fn):
    size = 0
    num_baches = len(dataloader)
    epoch_loss , epoch_correct= 0 ,0
    with torch.no_grad():
        model.eval()
        for i, (data_ , target_) in enumerate(dataloader):
            target_ = target_.type(torch.LongTensor)
            data_ , target_ = data_.to(device) , target_.to(device)  
            
            outputs = model(data_)
            
            loss = loss_fn(outputs , target_)
            
            epoch_loss = epoch_loss + loss.item()
            _,pred = torch.max(outputs , dim = 1)
            epoch_correct += torch.sum(pred == target_).item()
            
            size+= target_.shape[0]
    return epoch_correct/size  , epoch_loss / num_baches

EPOCHS = 50
logs = {"train_loss":[] , "train_acc":[] , "val_loss":[] , "val_acc":[]}

if os.path.exists('checkpoints') == False:
    os.mkdir('checkpoints')

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0001
momentum = .9
weight_decay = .1

optmizer = torch.optim.AdamW(model.parameters() , lr = learning_rate)

lr_milestones = [7 , 14, 21 , 28 , 35]
multi_step_lr_scheduler = lr_scheduler.MultiStepLR(optmizer ,
                                                   milestones=lr_milestones,
                                                   gamma = .1)

#Early stopping parameters
patience = 8
counter = 0
best_loss = np.inf

model.to(device)

for epoch in tqdm(range(EPOCHS)):
    train_acc , train_loss = train(train_loader ,
                                   model ,
                                   criterion ,
                                   optmizer ,
                                   multi_step_lr_scheduler)
    val_acc , val_loss = test(val_loader , model , criterion)
    print(f'epoch:{epoch} \
    train_loss = {train_loss:.4f} , train_acc:{train_acc:.4f} \
    val_loss = {val_loss:.4f} , val_acc:{val_acc:.4f} \
    learning rate: {optmizer.param_groups[0]["lr"]}')
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    logs['val_loss'].append(val_loss)
    logs['val_acc'].append(val_acc)
    
    if val_loss < best_loss:
        counter = 0
        best_loss = val_loss
        torch.save(model.state_dict() , "checkpoints\\best.pth")
    else:
        counter+=1
    if counter >= patience:
        print("Early stop !")
        break

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(logs['train_loss'],label='Train_Loss')
plt.plot(logs['val_loss'],label='Validation_Loss')
plt.title('Train_Loss & Validation_Loss',fontsize=20)
plt.legend()
plt.subplot(1,2,2)
plt.plot(logs['train_acc'],label='Train_Accuracy')
plt.plot(logs['val_acc'],label='Validation_Accuracy')
plt.title('Train_Accuracy & Validation_Accuracy',fontsize=20)
plt.legend()
plt.show()
