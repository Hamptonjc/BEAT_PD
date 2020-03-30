import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from tensorboard import notebook
from argparse import Namespace
import torch.nn as nn
import torchvision as tv
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils import data
from torch.utils.data import DataLoader

class LightningVGG16(pl.LightningModule):

    def __init__(self, hparams, train_list, val_list, label_class):
        super(LightningVGG16, self).__init__()
        self.vgg16 = VGG16
        self.learning_rate = hparams.learning_rate
        self.train_batch_size = hparams.train_batch_size
        self.val_batch_size = hparams.val_batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.label_class = label_class
        self.PredictionsDf = pd.DataFrame(columns=['measurement_id', 'subject_id', 'actual','predicted'])
    def forward(self, x):
        x = self.vgg16(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.vgg16.classifier[6].parameters(), lr=self.learning_rate)
    
    def training_step(self, train_batch, batch_idx):
        batch_corr = 0
        x, y = train_batch
        logits = self.forward(x[0])
        loss =  F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.train_batch_size) * 100)
        train_logs = {'training Loss': loss, 'Training Accuracy': acc, 'Number Correct in Training Batch': batch_corr}
        return {'loss': loss, 'Correct': batch_corr, 'acc': acc, 'log': train_logs}
    
    def validation_step(self, val_batch, batch_idx):
        batch_corr = 0
        x, y = val_batch
        logits = self.forward(x[0])
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.val_batch_size) * 100)
        pred = pred.cpu().tolist()
        pred = sum(pred, [])
        self.PredictionsDf = self.PredictionsDf.append(
            pd.DataFrame({'measurement_id':list(x[1]),'subject_id':list(x[2].cpu().tolist()), 'actual':y.tolist(),
                          'predicted':pred}),ignore_index=True)
        val_logs = {'Validation Loss': loss, 'Validation Accuracy': acc, 'Number Correct in Validation Batch': batch_corr}
        return {'val_loss': loss, 'val_acc': acc, 'log': val_logs}
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_score = float(BEATPDscoring(
            self.PredictionsDf)[r'$\frac{\sqrt{n_k} {MSE}_k}{\sum_{k=1}^N \sqrt{n_k} }$'].values.mean())
        tensorboard_logs = {'Average Validation Loss': avg_loss,
                            'Average Validation Accuracy': avg_acc, 'Average Validation BEAT_PD Score':avg_score}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}
        
    def prepare_data(self):
        self.prepped_trainset = Torch_Dataset(self.train_list, label_class=self.label_class)
        self.prepped_valset = Torch_Dataset(self.val_list, label_class=self.label_class)
        
    def train_dataloader(self):
        return DataLoader(self.prepped_trainset,batch_size=self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.prepped_valset,batch_size=self.val_batch_size, shuffle=True)