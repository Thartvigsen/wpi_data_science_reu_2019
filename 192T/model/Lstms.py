import torch
import torch.nn as nn
import numpy as np 

class LSTMs(nn.Module):

    def __init__(self, hidden_dim, output_dim, input_dim, N_LAYERS, batch_size, LAMBDA):

        super(LSTMs, self).__init__()

        self.HIDDEN_DIM = hidden_dim
        self.N_LAYERS = N_LAYERS
        self.BATCH_SIZE = batch_size
        self.LAMBDA = LAMBDA

        self.LSTM = nn.LSTM(input_dim, self.HIDDEN_DIM, N_LAYERS)
        self.out = nn.Linear(self.HIDDEN_DIM, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence):
        state = (torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM), torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM))
        sequence = torch.transpose(sequence,0,1)
        hidden, state = self.LSTM(sequence, state)
        output = self.out(hidden)
        self.prediction = self.sigmoid(output)
        #self.output = self.out(hidden[-1])
        #prediction = self.sigmoid(self.output)
        return self.prediction[-1]

    def applyLoss(self, predictions, labels):

        criterion=nn.CrossEntropyLoss()
        loss_c = criterion(predictions, labels)
        
        prediction_diffs = []
        for i in range(self.prediction.shape[0]):
            if i == 0:
                prediction_diffs.append(self.prediction[0]-self.prediction[0])
            if i > 0:
                prev_max, _ = self.prediction[:i].max(0)
                prediction_diffs.append(prev_max - self.prediction[i])
        prediction_diffs = torch.stack(prediction_diffs, 0).clamp(0)
        loss_s = torch.sum(prediction_diffs)

        loss = loss_c + self.LAMBDA*loss_s
        return loss, loss_s  













