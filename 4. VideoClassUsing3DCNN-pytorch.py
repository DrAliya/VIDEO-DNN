#!/usr/bin/env python
# coding: utf-8

# In[72]:


import sys
from pytorchvideo.data import LabeledVideoDataset
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('pytorchvideo')


# In[73]:


non = glob('NonViolence/*')
vio = glob('Violence/*')
label=[0]*len(non)+[1]*len(vio)


# In[77]:


df=pd.DataFrame(zip(vio+non, label), columns=['file', 'label'])
print(df)
print(len(non))
print(len(vio))


# In[80]:


df.head()


# In[130]:


from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(df,test_size=0.2,shuffle = True)


# In[131]:


len(train_df), len(val_df)


# #augmentation process

# In[106]:


from pytorchvideo.data import LabeledVideoDataset,make_clip_sampler,labeled_video_dataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute
)


# In[107]:


from torchvision.transforms import (
    Compose,
    #Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)


# In[108]:


from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)


# In[109]:


video_transform=Compose([
    ApplyTransformToKey(key='video',
    transform = Compose([
        UniformTemporalSubsample(20),
        #Lambda(lambda x:x/255),
        Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225)),
        RandomShortSideScale(min_size=248, max_size=256),
        CenterCropVideo(224),
        RandomHorizontalFlip(p=0.5)
    ]),
    ),
])


# In[110]:


print(train_df[0:5])


# In[111]:


import os

notebook_path = os.path.abspath("__file__")
print("Notebook path:", notebook_path)


# In[112]:


import pandas as pd

train_df.to_csv('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/train1_data.csv', index=False)
val_df.to_csv('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/val1_data.csv', index=False)


# In[113]:


from torch.utils.data import DataLoader

# Call the labeled_video_dataset method with the file path
train_dataset = labeled_video_dataset('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/train_data', clip_sampler=make_clip_sampler('random', 2),
                                      transform=video_transform, decode_audio=False)
loader1=DataLoader(train_dataset,batch_size=5,num_workers=0,pin_memory=False)
#uniform sampler takes 2 sec frames in the whole video sequentially


# In[114]:


print(train_dataset)
print(loader1)


# In[115]:


test_dataset = labeled_video_dataset('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/test_data', clip_sampler=make_clip_sampler('random', 2),
                                      transform=video_transform, decode_audio=False)
loader2=DataLoader(test_dataset,batch_size=5,num_workers=0,pin_memory=False)


# In[116]:


print(test_dataset)
print(loader2)


# In[117]:


import torch.nn as nn
import torch
import torch.utils.data
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import lr_scheduler 
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import torchmetrics
import torch.optim as optim
import pytorch_lightning as pl


# In[118]:


video_model=torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)


# In[119]:


video_model


# In[141]:


class OurModel(pl.LightningModule):
    def __init__ (self):
        super(OurModel, self).__init__()
        #model architecture
        self.video_model=torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(400,1)
        self.lr=1e-3
        self.batch_size=4
        self.numworker=4
        #6,4,8
        #evaluation metric
        self.metric=torchmetrics.Accuracy(task='binary')
        #loss function
        self.criterion=nn.BCEWithLogitsLoss()
    
    def forward(self, x, target):
        x=self.video_model(x)
        x=self.relu(x)
        x=self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        video,label=batch['video'], batch['label']
        out=self.forward(video)
        loss=self.criterion(out)
        metric=self.metric(out,label.to(torch.int64))
        #out=self(video)
        return {'loss':loss, 'metric':metric.detach()}
    
    def train_dataloader(self):

        dataset=labeled_video_dataset('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/train_data',clip_sampler=make_clip_sampler('random', 2),
                                    transform=video_transform, decode_audio=False)
        loader=DataLoader(dataset,batch_size=self.batch_size,num_workers=0,pin_memory=True)
        return loader
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
        return [optimizer], [lr_scheduler]
    

    
    def on_train_epoch_end(self, outputs):
        loss=torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        #stack them, mean and CPU and numpy
        metric=torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('training_loss', loss)
        self.log('training_metric', metric)

    def val_dataloader(self):
        dataset=LabeledVideoDataset('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/test_data',clip_sampler=make_clip_sampler('random', 2),
                                    transform=video_transform, decode_audio=False)
        loader=DataLoader(dataset,batch_size=self.batch_size,num_workers=0,pin_memory=True)
        return loader
    
    def validation_step(self,batch,batch_idx):
        video,label=batch['video'], batch['label']
        out=self.forward(video)
        loss=self.criterion(out)
        metric=self.metric(out,label.to(torch.int64))
        #out=self(video)
        return {'loss':loss, 'metric':metric.detach()}
    
    def on_validation_epoch_end(self,outputs):
        loss=torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        #stack them, mean and CPU and numpy
        metric=torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('val_loss', loss)
        self.log('val_metric', metric)
        
    def test_dataloader(self):
        dataset=LabeledVideoDataset('/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/test_data.csv',clip_sampler=make_clip_sampler('random', 2),
                                    transform=video_transform, decode_audio=False)
        loader=DataLoader(dataset,batch_size=self.batch_size,num_workers=self.numworker,pin_memory=True)
        return loader
    
    def test_step(self,batch,batch_idx):
        video,label=batch['video'], batch['label']
        out=self(video)
        #loss=self.criterion(out)
       # metric=self.metric(out,label.to(torch.int64))
        #out=self(video)
        return {'label':label, 'pred':out.detach()}
    
    def on_test_epoch_end(self,outputs):
        label=torch.cat(x['label'] for x in outputs).cpu().numpy()
        pred=torch.cat(x['pred'] for x in outputs).cpu().numpy()
        pred=np.where(pred>0.5,1,0)
        print(classfication_report(label,pred))

       # loss=torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        #stack them, mean and CPU and numpy
       # metric=torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
       # self.log('test_loss', loss)
       # self.log('test_metric', metric)


# In[142]:


m=OurModel()


# In[143]:


m.train_dataloader()


# In[144]:


m.test_dataloader()


# In[145]:


m.val_dataloader()


# In[146]:


checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints',
                                     filename='file',save_last=True)
lr_monitor = LearningRateMonitor(logging_interval='epoch')


# In[147]:


#Total 10 epochs, at 5 model is improving, at 7 model is interrupted, 
#if we want to resume at epoch 7

#If save_last = True means the model will resume at 5


# In[148]:


model=OurModel()
seed_everything(0)
trainer = Trainer(max_epochs=1,
                 accelerator='cpu', devices=1,
                 precision=16,
                 accumulate_grad_batches=2,
                 enable_progress_bar = False,
                 num_sanity_val_steps=0,
                 callbacks=[lr_monitor,checkpoint_callback],
                 limit_train_batches=5,
                 limit_val_batches=1,)
                 


# In[ ]:


trainer.fit(model)

