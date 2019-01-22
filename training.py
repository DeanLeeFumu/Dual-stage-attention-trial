

import sys

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from Model import encoder,decoder

from sklearn.preprocessing import StandardScaler
import matplotlib

from Earlystop_torch import EarlyStopping
# matplotlib.use('Agg')
get_ipython().magic(u'matplotlib inline')

import datetime as dt, pandas as pd, matplotlib.pyplot as plt, numpy as np

import utility as util
global logger


util.setup_log()
logger = util.logger


use_cuda = torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)

lstm_layers =3
path_dir = "D:/Python/DARNN/model monitor/"


## ONLY FOR TRAINS
def preprocess_data(dat, col_names):
    if type(col_names) is not tuple:
        print('colnames must be a tuple!')
        return None,None
    

    mask = np.ones(dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    
    dat = np.array(dat)
    
    feats = dat[:, mask]
    targs = dat[:, ~mask]
    scalefeats = StandardScaler().fit(feats)
    scaletargs = StandardScaler().fit(targs)
    feats = scalefeats.transform(feats)
    targs = scaletargs.transform(targs)

    return (feats, targs), (scalefeats,scaletargs)





# Train the model
class da_rnn:
    def __init__(self, file_data,  logger, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10,targ_col = 'NDX',
                 learning_rate = 0.01, batch_size = 128,lstm_layers = 2, parallel = True, debug = False, early_stop = 10):
        self.T = T
        dat = pd.read_csv(file_data, nrows = 200 if debug else None)
        self.targ_col = targ_col
        self.dat ,self.scalers = preprocess_data(dat,(self.targ_col,))
        self.scaler_train = self.scalers[0]
        self.scaler_targ = self.scalers[1]

        self.logger = logger
        self.logger.info("Shape of data: %s.\nMissing in data: %s.", dat.shape, dat.isnull().sum().sum())
        self.earlystop = early_stop
        self.X = self.dat[0]
        self.y = self.dat[1].reshape(-1)
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        print(self.X.shape,self.y.shape)
        
        if self.earlystop:
            self.stopper = EarlyStopping(mode='min', min_delta=0, patience = self.earlystop, percentage=False)

        self.encoder = encoder(input_size = self.X.shape[1], hidden_size = encoder_hidden_size, T = T,
                              logger = logger,layers = self.lstm_layers).cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T,
                               layers = self.lstm_layers).cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        # self.learning_rate = learning_rate

        self.train_size = int(self.X.shape[0] * 0.7)
        #self.y = self.y - np.mean(self.y[:self.train_size]) # Question: why Adam requires data to be normalized?
        self.logger.info("Training size: %d.", self.train_size)
        

    def train(self, n_epochs = 10):
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        logger.info("Iterations per epoch: %3.3f ~ %d.", self.train_size * 1. / self.batch_size, iter_per_epoch)
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()

        n_iter = 0
        self.stop = False
        #learning_rate = 1.

        for i in range(n_epochs):
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T - 1), :]
                    y_history[k, :] = self.y[batch_idx[k] : (batch_idx[k] + self.T - 1)]

                loss = self.train_iteration(X, y_history, y_target)#.cpu()
                self.iter_losses[int(i * iter_per_epoch + j / self.batch_size)] = loss#.numpy()
                #if (j / self.batch_size) % 50 == 0:
                #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / self.batch_size, loss)
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                '''
                if learning_rate > self.learning_rate:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * .9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * .9
                    learning_rate *= .9
                '''


            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            #early stop 
            self.stop = self.stopper.step(self.epoch_losses[i])
            #encoder & decoder saver
            self.stopper.savemodel(self.epoch_losses[i],self.encoder,'encoder')
            self.stopper.savemodel(self.epoch_losses[i],self.decoder,'decoder')


            if i % 10 == 0:
                self.logger.info("Epoch %d, loss: %3.3f.", i, self.epoch_losses[i])

            if i % 10 == 0:
                y_train_pred = self.predict(on_train = True)
                y_test_pred = self.predict(on_train = False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y) + 1), y_test_pred, label = 'Predicted - Test')
                plt.legend(loc = 'upper left')
                plt.show()
                
            if self.stop:
                self.logger.info("Early_stop Epoch at %d,best loss: %3.3f.", i, self.epoch_losses[i-9])
                break

    def train_iteration(self, X, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
        y_pred = self.decoder(input_encoded, Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))

        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())
        loss = self.loss_func(y_pred,y_true[:,np.newaxis])
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # if loss.data[0] < 10:
        #     self.logger.info("MSE: %s, loss: %s.", loss.data, (y_pred[:, 0] - y_true).pow(2).mean())

        return loss.item()


    def predict(self, on_train = False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred
    


if __name__ == '__main__':
    
    #Debug mode = T
    io_dir = '~/nasdaq'
    model = da_rnn(file_data = 'D:/Python/DARNN/nasdaq100/small/nasdaq100_padding.csv'.format(io_dir),
                   debug = True, logger = logger, parallel = False,
                   early_stop = 100,
                   lstm_layers =3,
                   learning_rate = .001)
    
    
    model.train(n_epochs = 100)
    
    
    
    #load the best weights
    model.encoder.load_state_dict(torch.load(path_dir+'encoder_checkpoint.pt'))
    model.decoder.load_state_dict(torch.load(path_dir+'decoder_checkpoint.pt'))
    
    y_pred = model.predict()
    
    #training losses
    plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.show()
    
    plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.show()
    
    
    #check prediction
    
    
    plt.figure()
    plt.plot(y_pred, label = 'Predicted')
    plt.plot(model.y[model.train_size:], label = "True")
    plt.legend(loc = 'upper left')
    plt.show()
    
    ##Before scale
    
    #y_pred = model.scaler_targ.inverse_transform(y_pred)
    #plt.figure()
    #plt.plot(y_pred, label = 'Predicted')
    #plt.plot(model.scaler_targ.inverse_transform(model.y[model.train_size:]), label = "True")
    #plt.legend(loc = 'upper left')
    #plt.show()
