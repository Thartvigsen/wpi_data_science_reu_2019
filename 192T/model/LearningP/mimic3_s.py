'''
This code is essentially the "main" function. It calls the functions that actually load in the data, create the model, and perform evaluation.
'''

import torch
import torch.nn as nn
from data.dataLoader import dataLoader
from LSTMs import LSTMs
from trainRun import trainRun
from testRun import testRun
from params import params
from runModel import runModel

import sys

params = params() # This calls a class that handles the parameters of the model 

dataObj = dataLoader("time_series.pt", "labels.pt", "masks.pt", "diffs.pt", params.BATCH_SIZE) # Load in the data

train_loader, validation_loader, test_loader = dataObj.train_ix, dataObj.validation_ix, dataObj.test_ix # Split the data into train, validation, test

model = LSTMs(params.HIDDEN_DIMENSION, params.N_CLASSES, params.N_FEATURES, params.N_LAYERS, params.BATCH_SIZE, params.DROPOUT, params.LAMBDA) # Create the model

runModel(model, train_loader, validation_loader, test_loader, params, "results/"+sys.argv[2]+"loss/"+sys.argv[1]+".txt", "results/"+sys.argv[2]+"auc/"+sys.argv[1]+".txt", "results/"+sys.argv[2]+"loss_s/"+sys.argv[1]+".txt", "results/"+sys.argv[2]+"testLoss.txt", "results/"+sys.argv[2]+"testAUC.txt","results/"+sys.argv[2]+"testLoss_s.txt",NUM_ITER = 10) # Run and evaluate the model
