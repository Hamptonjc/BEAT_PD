import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type('torch.FloatTensor').cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type('torch.FloatTensor').cuda())

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), -1, 4).cuda())
        output = self.linear(lstm_out[-1].view(-1, 4))
        return output

