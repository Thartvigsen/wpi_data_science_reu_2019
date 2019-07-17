'''
This code actually runs the model. It first calls trainRun, which trains the model and tests using validation set. Then it tests the model. Will train model NUM_ITER times and print the average AUC
'''

from trainRun import trainRun
from testRun import testRun

def runModel(model, train_loader, validation_loader, test_loader, params, NUM_ITER):

    params.setOptimizer(model)
        
    model, params.optimizer = trainRun(model, train_loader, validation_loader, params.scheduler, params.optimizer, params.BATCH_SIZE, params.N_EPOCHS, params.criterion)

    auc = testRun(model, test_loader)

    print('Total AUC: {}'.format(auc))
