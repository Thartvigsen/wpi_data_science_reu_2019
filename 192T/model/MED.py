import torch
import torch.nn as nn
import numpy as np 
import time

class MED(nn.Module):

    def __init__(self, hidden_dim, output_dim, input_dim, N_LAYERS, batch_size, DROPOUT, LAMBDA):

        super(MED, self).__init__()

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
        return self.prediction

    def applyLoss(self, predictions, labels):

        criterion=nn.functional.binary_cross_entropy
        loss_c = criterion(predictions, labels)

        loss_s = 0

        start = time.time()
        predictionsNp = self.prediction.detach().numpy()
        for i in range(predictionsNp.shape[1]):
                for j in range(predictionsNp.shape[2]):
                    if(labels[i][j]==1):
                        b = True
                    else:
                        b = False
                    
                    if(b):
                        prev_max = 0
                        for k in range(predictionsNp.shape[0]):
                            if k == 0:
                                prev_max = predictionsNp[k][i][j]
                            else:
                                diff=(prev_max-predictionsNp[k][i][j])
                                loss_s+=diff
                                if(diff<0):
                                    prev_max = predictionsNp[k][i][j]
                    else:
                        prev_min = 0
                        for k in range(predictionsNp.shape[0]):
                            if k == 0:
                                prev_min = predictionsNp[k][i][j]
                            else:
                                diff=(predictionsNp[k][i][j]-prev_min)
                                loss_s+=diff
                                if(diff<0):
                                    prev_min = predictionsNp[k][i][j]
        end = time.time()
        print("2nd: ",loss_s)
        print("time elapsed: ", (end-start))
                  
           
        loss = loss_c + self.LAMBDA*loss_s
        return loss, loss_s













