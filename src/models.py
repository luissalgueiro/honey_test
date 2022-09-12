import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_toolbelt import losses as L
import matplotlib.pyplot as plt
import matplotlib 
import os
from torchmetrics import IoU, MeanSquaredError
import numpy as np
from torchmetrics import F1
import random
from typing import List
from collections import OrderedDict
device='cuda' if torch.cuda.is_available() else 'cpu'
# from kornia.filters import sobel, spatial_gradient, laplacian
# matplotlib.use("Agg")



class PLModel(pl.LightningModule):
    def __init__(self,\
        model,\
        config=None,\
        # class_weights,
        ):
        super().__init__( )
        self.model = model
        # self.class_weights=class_weights.to('cuda')
        self.focal_loss = L.FocalLoss(alpha=0.25, gamma=2)
        self.f1_score = F1(num_classes=12, average='weighted')
        # self.class_weights
        self.save_hyperparameters(config)
        
        
    def forward(self, x):
        return self.model(x)
    def predict(self, x):
        with torch.no_grad():
            y_hat = self(x)
            return torch.argmax(y_hat, axis=1)
    def compute_loss_and_metrics(self, batch):
        x, y = batch['image'], batch['target']
        # print(f'X: {x.shape} \t Y: {y.shape}')
        y_hat = self(x)
        # print(f'Output: {y_hat.shape}')
        # loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        loss = self.focal_loss(y_hat, y)
        # acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.shape[0]
        # y1 = y.detach().cpu().numpy()
        # print(y1.shape)
        y_hat1 = torch.argmax(y_hat, axis=1)
        # y_hat1 = y_hat1.detach().cpu().numpy()
        # print(y_hat1.shape)
        f1w = self.f1_score(y, y_hat1) # , average='weighted')
        return loss, f1w
    def training_step(self, batch, batch_idx):
        loss, f1w = self.compute_loss_and_metrics(batch)
        self.log('train_loss', loss)
        self.log('train_F1w', f1w, prog_bar=True)
        #print(f'Training_step: loss> {loss} acc:{acc}')
        return {'loss':loss,'f1w':torch.tensor(f1w)}
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_train_f1w  = torch.stack([x['f1w'] for x in outputs]).mean()
        # train_epoch_loss_CE.append(avg_train_loss.item())
        # train_epoch_acc_CE.append(avg_train_f1w.item())
        #print(f'Epoch {self.current_epoch} TrainLOSS:{avg_train_loss} TrainACC:{avg_train_acc}  ')
    def validation_step(self, batch, batch_idx):
        loss, f1w = self.compute_loss_and_metrics(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1w', f1w, prog_bar=True)
        return {'val_loss': torch.tensor(loss.item()), 'val_f1w': torch.tensor(f1w)}
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_f1w  = torch.stack([x['val_f1w'] for x in outputs]).mean()
        self.log('EarlyStop_Log', avg_val_loss.detach(), on_epoch=True, sync_dist=True)
        self.log('avg_val_f1w', avg_val_f1w.detach(), on_epoch=True, sync_dist=True)
        # val_epoch_loss_CE.append(avg_val_loss.item())
        # val_epoch_acc_CE.append(avg_val_f1w.item())
        #print(f'VAL-Epoch {self.current_epoch} LOSS:{avg_val_loss} ACC:{avg_val_acc} ')
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=5e-5  )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                    T_0=10,
                                                                    T_mult=1,
                                                                    eta_min=1e-8,
                                                                    verbose=True,
                                                                    )

        # lr_scheduler = {'scheduler': MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.5,),'interval': 'epoch','frequency':1}
        return [optimizer], [lr_scheduler]