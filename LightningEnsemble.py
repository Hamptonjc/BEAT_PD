
# Imports
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


torch.manual_seed(314)

#Edit VGG16 architechture for use with CIS-PD Data.
CIS_VGG16 = models.vgg16(pretrained=True)
for param in CIS_VGG16.parameters():
    param.requires_grad = False
CIS_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,5))

#Edit VGG16 architechture for use with REAL-PD Data.
REAL_ON_OFF_VGG16 = models.vgg16(pretrained=True)
for param in REAL_ON_OFF_VGG16.parameters():
    param.requires_grad = False
REAL_ON_OFF_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,2))

#Edit VGG16 architechture for use with REAL-PD Data.
REAL_DYSK_VGG16 = models.vgg16(pretrained=True)
for param in REAL_DYSK_VGG16.parameters():
    param.requires_grad = False
REAL_DYSK_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,3))

#Edit VGG16 architechture for use with REAL-PD Data.
REAL_TREMOR_VGG16 = models.vgg16(pretrained=True)
for param in REAL_TREMOR_VGG16.parameters():
    param.requires_grad = False
REAL_TREMOR_VGG16.classifier[6] = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),
                                    nn.Linear(512,4))



class LightningEnsemble(pl.LightningModule):

    '''
    dataset_name: 'CIS' or 'REAL'

    hparams: Namespace(**{'learning_rate':_ , 'train_batch_size':_ , 'val_batch_size':_ })

    train_list: python list of preprocessed training data from Dataset class.

    val_list: python list of preprocessed validation data from Dataset class.

    Label_class: 'on_off', 'dyskinesia', or 'tremor'

    '''

    def __init__(self, dataset_name, hparams, train_list, val_list, label_class, path_to_null_preds, gpus):
        super(LightningEnsemble, self).__init__()
        self.dataset_name = dataset_name
        self.learning_rate = hparams.learning_rate
        self.train_batch_size = hparams.train_batch_size
        self.val_batch_size = hparams.val_batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.label_class = label_class
        self.gpus = gpus
        self.null_preds = pd.read_csv(path_to_null_preds, index_col='subject_id')#.rename(index=str)
        self.PredictionsDf = pd.DataFrame(columns=['measurement_id', 'subject_id', 'actual','predicted'])
        #Specify correct models to use given the data set (CIS or REAL).
        if dataset_name == 'CIS':
            self.vgg16 = CIS_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=5, gpus=self.gpus)
            self.classifier = nn.Linear(11, 5)

        elif dataset_name == 'REAL' and label_class == 'on_off':
            self.vgg16 = REAL_ON_OFF_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=2, gpus=self.gpus)
            self.classifier = nn.Linear(5, 2)

        elif dataset_name == 'REAL' and label_class == 'dyskinesia':
            self.vgg16 = REAL_DYSK_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=3, gpus=self.gpus)
            self.classifier = nn.Linear(7, 3)

        elif dataset_name == 'REAL' and label_class == 'tremor':
            self.vgg16 = REAL_TREMOR_VGG16
            self.lstm = LSTM(input_dim=4, hidden_dim=4, batch_size=self.train_batch_size, output_dim=4, gpus=self.gpus)
            self.classifier = nn.Linear(9, 4)
        
        else:
            raise TypeError('Specifiy dataset_name and label_class')

    def forward(self, x):
        if self.gpus:
            try: #Training/validation
                spec_output = self.vgg16(x[0])
                ts_output = self.lstm(x[6].permute(1,0,2).type('torch.FloatTensor'))
                if self.dataset_name == 'CIS':
                    null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[tuple(x[2].cpu().numpy().astype('str')), self.label_class]),dim=0).reshape(-1,1).cuda()
                else:
                    null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[x[2], self.label_class]),dim=0).reshape(-1,1).cuda()
                final_output = self.classifier(torch.cat((spec_output, ts_output, null_pred),dim=1))
            except: # Predicting
                spec_output = self.vgg16(x[0])
                ts_output = self.lstm(x[1].permute(1,0,2).type('torch.FloatTensor'))
                null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[str(x[2]), self.label_class]),dim=0).reshape(-1,1).cuda()
                final_output = self.classifier(torch.cat((spec_output, ts_output, null_pred),dim=1))
        else:
            try: #Training/validation
                spec_output = self.vgg16(x[0])
                ts_output = self.lstm(x[6].permute(1,0,2).type('torch.FloatTensor'))
                if self.dataset_name == 'CIS':
                    null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[tuple(x[2].cpu().numpy().astype('str')), self.label_class]),dim=0).reshape(-1,1)
                else:
                    null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[x[2], self.label_class]),dim=0).reshape(-1,1)
                final_output = self.classifier(torch.cat((spec_output, ts_output, null_pred),dim=1))
            except: # Predicting
                spec_output = self.vgg16(x[0])
                ts_output = self.lstm(x[1].permute(1,0,2).type('torch.FloatTensor'))
                null_pred = torch.unsqueeze(torch.tensor(self.null_preds.loc[str(x[2]), self.label_class]),dim=0).reshape(-1,1)
                final_output = self.classifier(torch.cat((spec_output, ts_output, null_pred),dim=1))
        return final_output
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(list(self.vgg16.classifier[6].parameters())+ list(self.lstm.parameters()) +
            list(self.classifier.parameters()), lr=self.learning_rate)
        return optim

    def training_step(self, train_batch, batch_idx):
        batch_corr = 0
        x, y = train_batch
        logits = self.forward(x)
        loss =  F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.train_batch_size) * 100)
        train_logs = {'training Loss': loss, 'Training Accuracy': acc, 'Number Correct in Training Batch': batch_corr}
        return {'loss': loss, 'Correct': batch_corr, 'acc': acc, 'log': train_logs}
    
    def validation_step(self, val_batch, batch_idx):
        batch_corr = 0
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True)
        batch_corr += pred.eq(y.view_as(pred)).sum().item()
        acc = torch.tensor((batch_corr/self.val_batch_size) * 100)
        pred = pred.cpu().tolist()
        pred = sum(pred, [])
        try: # For BEATPD scoring
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
        score = float(BEATPDscoring(self.PredictionsDf).iloc[:,-1].sum())# For BEATPD scoring
        self.PredictionsDf = pd.DataFrame(columns=['measurement_id', 'subject_id', 'actual','predicted']) #Reset prediction df
        tensorboard_logs = {'Average Validation Loss': avg_loss,
                            'Average Validation Accuracy': avg_acc, 'Average Validation BEAT_PD Score':score}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}


    def prepare_data(self):
        '''
        Prep data with Torch Dataset. Converts data to tensors etc.
        '''
        self.prepped_trainset = Torch_Dataset(self.train_list, label_class=self.label_class)
        self.prepped_valset = Torch_Dataset(self.val_list, label_class=self.label_class)
        
    def train_dataloader(self):
        return DataLoader(self.prepped_trainset,batch_size=self.train_batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.prepped_valset,batch_size=self.val_batch_size, shuffle=True, drop_last=True)



