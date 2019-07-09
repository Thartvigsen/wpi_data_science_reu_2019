'''
This code is essentially the "main" function. It actually loads in the data, creates the model, and performs evaluation
'''

import torch 
import torch.nn as nn
from turingFinal import dataLoader
from rnn import RNN
from trainRun import trainRun
from testRun import testRun
from runModel import runModel
from params import params


params = params() # This calls a class that handles the parameters of the model


dataObj = dataLoader("time_series.pt", "labels.pt", "masks.pt", params.BATCH_SIZE) # Load in the data


train_loader, validation_loader, test_loader = dataObj.train_ix, dataObj.validation_ix, dataObj.test_ix # Split the data into train, validation, test
        

model = RNN(params.HIDDEN_DIMENSION, params.N_CLASSES, params.N_FEATURES, params.N_LAYERS, params.BATCH_SIZE, params.DROPOUT) # Create the model


runModel(model, train_loader, validation_loader, test_loader, params, NUM_ITER = 10) # Run and evaluate the model. NUM_ITER is the number of times the model runs and then averages results together.
