#This code runs the model through training and validation data

import torch
import sklearn
from sklearn import metrics
import numpy as np
import os
from testRunMed import testRun

def trainRun(model, train_loader, test_loader, validation_loader, scheduler, optimizer, BATCH_SIZE, N_EPOCHS, criterion, fileName1, fileName2, fileName3, fileName4, fileName5, fileName6, LAMBDA):

    print("LAMBDA = ", model.LAMBDA)

    f1 = open(fileName1, "a+")
    f2 = open(fileName2, "a+")
    f3 = open(fileName3, "a+")
     
    for epoch in range(N_EPOCHS):
           
            scheduler.step()
    
            for i, (time_series, labels) in enumerate(train_loader):

                model.zero_grad()

                predictions = model.forward(time_series)

                loss, _ = model.applyLoss(predictions, labels)

                loss.backward()

                optimizer.step()

            count = 0
            aucTotal = 0
            lossTotal = 0
            loss_s_total = 0
 
            for i, (time_series, labels) in enumerate(validation_loader):

                model.zero_grad()

                predictions = model.forward(time_series)

                loss, loss_s = model.applyLoss(predictions, labels)

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
            loss_sFinal = loss_s_total/count

            print('Epoch [{}/{}], Validation AUC: {}'.format(epoch+1, N_EPOCHS, auc))       
            print('Epoch [{}/{}], Validation Loss: {}'.format(epoch+1, N_EPOCHS, lossFinal))

            f1.write('%f,' % lossFinal)
            f2.write('%f,' % auc)
            f3.write('%f,' % loss_sFinal)

    f1.seek(f1.tell() -1, 0)
    f1.truncate()


    f2.seek(f2.tell() -1, 0)
    f2.truncate()

    f3.seek(f3.tell() -1, 0)
    f3.truncate()
    
    return model, optimizer 
