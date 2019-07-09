#This defines the LSTM that is used on the MIMIC-III data. Uses Sigmoid activation function

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS, BATCH_SIZE, DROPOUT):
        super(RNN, self).__init__() #Set instance values
        self.N_LAYERS = N_LAYERS
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_DIMENSION = HIDDEN_DIMENSION
        self.N_CLASSES = N_CLASSES
    
        self.LSTM = torch.nn.LSTM(N_FEATURES, HIDDEN_DIMENSION, N_LAYERS, dropout=DROPOUT)
        self.out = torch.nn.Linear(HIDDEN_DIMENSION, N_CLASSES)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):

        state = (torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIMENSION), torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIMENSION))#Initialize state to 0
        
        X = torch.transpose(X, 0, 1)
        
        hidden, state = self.LSTM(X, state)
        output = self.out(hidden[-1])
        prediction = self.sigmoid(output)
        return prediction
