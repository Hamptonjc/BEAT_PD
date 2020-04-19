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
#import BEATPD_LSTM, Torch_Dataset


torch.manual_seed(314)
CIS_VGG16 = models.vgg16(pretrained=True)
for param in CIS_VGG16.parameters():
    param.requires_grad = False
CIS_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,5))

REAL_VGG16 = models.vgg16(pretrained=True)
for param in REAL_VGG16.parameters():
    param.requires_grad = False
REAL_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,2))



class LightningVGG16(pl.LightningModule):

    '''
    dataset_name: 'CIS' or 'REAL'

    hparams: Namespace(**{'learning_rate':_ , 'train_batch_size':_ , 'val_batch_size':_ })

    train_list: python list of preprocessed training data from Dataset class.

    val_list: python list of preprocessed validation data from Dataset class.

    Label_class: 'on_off', 'dyskinesia', or 'tremor'

    '''

    def __init__(self, dataset_name, hparams, train_list, val_list, label_class):
        super(LightningVGG16, self).__init__()
        self.learning_rate = hparams.learning_rate
        self.train_batch_size = hparams.train_batch_size
        self.val_batch_size = hparams.val_batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.label_class = label_class
        self.PredictionsDf = pd.DataFrame(columns=['measurement_id', 'subject_id', 'actual','predicted'])
        if dataset_name == 'CIS':
            self.vgg16 = CIS_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=5)
            self.classifier = nn.Linear(10, 5)

        elif dataset_name == 'REAL':
            self.vgg16 = REAL_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=2)
            self.classifier = nn.Linear(4, 2)
        else:
            raise TypeError('Specifiy CIS or REAL for dataset_name')

    def forward(self, x):
        spec = self.vgg16(x[0])
        ts = self.lstm(x[6].permute(1,0,2).type('torch.FloatTensor'))
        output = self.classifier(torch.cat([spec.view(-1), ts]))
        return output
    
    def configure_optimizers(self):
        vgg_optim = torch.optim.Adam(list(self.vgg16.classifier[6].parameters())+ list(self.lstm.parameters()) +
            list(self.classifier.parameters()), lr=self.learning_rate)
        #lstm_optim = torch.optim.Adam(self.lstm.parameters(), lr = self.learning_rate)
        #classifier_optim = torch.optim.Adam(self.classifier.parameters(), lr = self.learning_rate)
        #return [vgg_optim, lstm_optim, classifier_optim]
    #list(fc1.parameters()) + list(fc2.parameters())
        return vgg_optim

    def training_step(self, train_batch, batch_idx):
        batch_corr = 0
        x, y = train_batch
        logits = self.forward(x).unsqueeze(0)
        loss =  F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.train_batch_size) * 100)
        train_logs = {'training Loss': loss, 'Training Accuracy': acc, 'Number Correct in Training Batch': batch_corr}
        return {'loss': loss, 'Correct': batch_corr, 'acc': acc, 'log': train_logs}
    
    def validation_step(self, val_batch, batch_idx):
        batch_corr = 0
        x, y = val_batch
        logits = self.forward(x).unsqueeze(0)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.val_batch_size) * 100)
        pred = pred.cpu().tolist()
        pred = sum(pred, [])
        try:
            self.PredictionsDf = self.PredictionsDf.append(
                pd.DataFrame({'measurement_id':list(x[1]),'subject_id':list(x[2].cpu().tolist()), 'actual':y.tolist(),
                    'predicted':pred}),ignore_index=True)
        except:
            self.PredictionsDf = self.PredictionsDf.append(
                pd.DataFrame({'measurement_id':list(x[1]),'subject_id':list(x[2]), 'actual':y.tolist(),
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



