#This code runs the model through training and validation data

import torch
import sklearn
from sklearn import metrics
import numpy as np
import os
from testRun import testRun

def trainRun(model, train_loader, test_loader, validation_loader, scheduler, optimizer, BATCH_SIZE, N_EPOCHS, criterion, fileName1, fileName2, fileName3, fileName4):

    f1 = open(fileName1, "a+")
    f2 = open(fileName2, "a+")
     
    for epoch in range(N_EPOCHS):
           
            scheduler.step()
    
            for i, (time_series, labels) in enumerate(train_loader):

                model.zero_grad()

                predictions = model(time_series)

                loss = criterion(predictions, labels)

                loss.backward()

                optimizer.step()

            count = 0
            aucTotal = 0
            lossTotal = 0
 
            for i, (time_series, labels) in enumerate(validation_loader):

                model.zero_grad()

                predictions = model(time_series)

                loss = criterion(predictions, labels)

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
                lossTotal += loss/BATCH_SIZE    

            auc = aucTotal/count
            lossFinal = lossTotal/count

            print('Epoch [{}/{}], Validation AUC: {}'.format(epoch+1, N_EPOCHS, auc))       
            print('Epoch [{}/{}], Validation Loss: {}'.format(epoch+1, N_EPOCHS, lossFinal))

            f1.write('%f,' % lossFinal)
            f2.write('%f,' % auc)

            testRun(model, test_loader, BATCH_SIZE, criterion, fileName3, fileName4, epoch)

    f1.seek(f1.tell() -1, 0)
    f1.truncate()


    f2.seek(f2.tell() -1, 0)
    f2.truncate()
    
    return model, optimizer 
