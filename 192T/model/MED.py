import torch
import torch.nn as nn
import numpy as np 

class LSTMs(nn.Module):

    def __init__(self, hidden_dim, output_dim, input_dim, N_LAYERS, batch_size, DROPOUT, LAMBDA):

        super(LSTMs, self).__init__()

        self.HIDDEN_DIM = hidden_dim
        self.N_LAYERS = N_LAYERS
        self.BATCH_SIZE = batch_size
        self.LAMBDA = LAMBDA

        self.LSTM = nn.LSTM(input_dim, self.HIDDEN_DIM, N_LAYERS, dropout=DROPOUT)
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

        criterion=nn.functional.binary_cross_entropy
        loss_c = criterion(predictions, labels)
 
        patientLoss = []   

        for i in range(self.prediction.shape[1]):
            for j in range(self.prediction.shape[2]):
                prediction_diffs = []
                for k in range(self.prediction.shape[0]):
                    if k == 0:
                        prediction_diffs.append(0)
                    if(labels[i,j]==1):
                        prev_max, _ = self.prediction[:k,i,j].max(0)
                        prediction_diffs.append(prev_max - self.prediction[k, i, j])
                    else:
                        prev_min, _ = self.prediction[:k,i,j].min(0)
                        prediction_diffs.append(self.prediction[k,i,j]-prev_min)
                        
            patientLoss.append(sum(prediction_diffs))
        loss_s = sum(patientLoss)/1280

        loss = loss_c + self.LAMBDA*loss_s
        return loss, loss_s 













