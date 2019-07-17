#This code runs the model through training and validation data

import torch
import sklearn
from sklearn import metrics
import numpy as np

def trainRun(model, train_loader, validation_loader, scheduler, optimizer, N_EPOCHS, criterion):

    for epoch in range(N_EPOCHS):

        scheduler.step()

        for i, (time_series, labels) in enumerate(train_loader):

            model.zero_grad()

            predictions = model(time_series)

            loss = model.applyLoss(predictions, labels)

            loss.backward()

            optimizer.step()

        count = 0
        aucTotal = 0
        
        for i, (time_series, labels) in enumerate(validation_loader):

            model.zero_grad()

            predictions = model(time_series)

            count += 1
            
            arr = np.sum(labels.data.numpy(),axis=0)
            
            for i in range(len(arr)):
                if(arr[i]==0):
                    arr[i]=1
                    labelsT = torch.transpose(labels, 0, 1)
                    labelsT[i][0] = 1
                    labels = torch.transpose(labelsT, 0, 1)
                elif(arr[i]==model.BATCH_SIZE):
                    arr[i]=9
                    labelsT = torch.transpose(labels, 0, 1)
                    labelsT[i][0]=0
                    labels = torch.transpose(labelsT, 0, 1)
            arr = np.sum(labels.data.numpy(), axis = 0)
            aucTotal += metrics.roc_auc_score(labels, predictions.detach(), average="micro")

        auc = aucTotal/count
        print('Epoch [{}/{}], Validation AUC: {}'.format(epoch+1, N_EPOCHS, auc))       

    return model, optimizer 
