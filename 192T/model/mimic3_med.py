'''
This code is essentially the "main" function. It calls the functions that actually load in the data, create the model, and perform evaluation.
'''

import torch
import torch.nn as nn
from data.dataLoader import dataLoader
from LSTMs import LSTMs
from trainRunMed import trainRun
from testRunMed import testRun
from paramsMed import params
from runModelMed import runModel

import sys

params = params(float(sys.argv[4])) # This calls a class that handles the parameters of the model 

dataObj = dataLoader(sys.argv[3]+".pt", "labels.pt", "masks.pt", "diffs.pt", params.BATCH_SIZE) # Load in the data

train_loader, validation_loader, test_loader = dataObj.train_ix, dataObj.validation_ix, dataObj.test_ix # Split the data into train, validation, test

model = LSTMs(params.HIDDEN_DIMENSION, params.N_CLASSES, params.N_FEATURES, params.N_LAYERS, params.BATCH_SIZE, params.DROPOUT, params.LAMBDA) # Create the model

runModel(model, train_loader, validation_loader, test_loader, params,"results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/trainLoss/"+sys.argv[1]+".txt","results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/aucTrain/"+sys.argv[1]+".txt", "results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/trainLoss_s/"+sys.argv[1]+".txt", "results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/testLoss.txt", "results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/testAUC.txt", "results/"+sys.argv[2]+sys.argv[3]+"/"+sys.argv[4]+"/testLoss_s.txt",sys.argv[4]) # Run and evaluate the model
