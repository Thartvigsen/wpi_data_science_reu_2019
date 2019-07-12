'''
This code is essentially the "main" function. It calls the functions that actually load in the data, create the model, and perform evaluation.
'''

import torch
import torch.nn as nn
from data.dataLoader import dataLoader
from rnn import RNN
from trainRun import trainRun
from testRun import testRun
from params import params
from runModel import runModel
from performImpute import performImpute

params = params() # This calls a class that handles the parameters of the model 

dataObj = dataLoader("time_series.pt", "labels.pt", "masks.pt", "diffs.pt", params.BATCH_SIZE) # Load in the data

#performImpute(dataObj)

train_loader, validation_loader, test_loader = dataObj.train_ix, dataObj.validation_ix, dataObj.test_ix # Split the data into train, validation, test

model = RNN(params.HIDDEN_DIMENSION, params.N_CLASSES, params.N_FEATURES, params.N_LAYERS, params.BATCH_SIZE, params.DROPOUT) # Create the model

runModel(model, train_loader, validation_loader, test_loader, params, NUM_ITER = 10) # Run and evaluate the model
