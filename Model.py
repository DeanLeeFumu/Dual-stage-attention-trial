import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
lstm_layers =3
path_dir = "D:/Python/DARNN/model monitor/"


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T, logger, layers = lstm_layers):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.layers = layers

        self.logger = logger

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = layers)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)

    def forward(self, input_data):
        #print("input_data",input_data.size())
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # layers * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden[0,:,:].repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell[0,:,:].repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)
            #print("xsize: ",x.size())
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)) # (batch_size * input_size) * 1
            # Eqn. 9: Get attention weights
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=1) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            #print("weighted_input.unsqueeze(0)",weighted_input.unsqueeze(0).size())
            out, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            #("weighted_input %s hidden %s" % (weighted_input.size(),hidden.size())) # torch.Size([batch, column ]) torch.Size([layers, batch, hidden size])
            
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = out  #(Batch, Times,  Hidden size)
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(self.layers, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, layers = lstm_layers, out_feats=1):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.layers = layers


        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), 
                                         nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size,num_layers = self.layers)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden[0,:,:].repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell[0,:,:].repeat(self.T - 1, 1, 1).permute(1, 0, 2), 
                           input_encoded), dim = 2)
            # Eqn. 12-13: compute attention weights            
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T - 1),dim = 1) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
           # if t < self.T - 1:
            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(self.layers, x.size(0), self.decoder_hidden_size).zero_())

