'''
This code actually runs the model. It first calls trainRun, which trains the model and tests using validation set. Then it tests the model. Will train model NUM_ITER times and print the average AUC
'''

from trainRunMed import trainRun
from testRunMed import testRun

def runModel(model, train_loader, validation_loader, test_loader, params, fileName, fileName2, fileName3, fileName4, fileName5, fileName6, LAMBDA):

    params.setOptimizer(model)
        
    model, params.optimizer = trainRun(model, train_loader, test_loader, validation_loader, params.scheduler, params.optimizer, params.BATCH_SIZE, params.N_EPOCHS, params.criterion, fileName, fileName2, fileName3, fileName4, fileName5, fileName6, LAMBDA, params.ALPHA)

    auc = testRun(model, test_loader, params.BATCH_SIZE, params.criterion, fileName4, fileName5, fileName6, LAMBDA)
