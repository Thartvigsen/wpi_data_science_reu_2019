#This code runs the model through training and validation data

import torch
import sklearn
from sklearn import metrics
import numpy as np

def trainRun(model, train_loader, validation_loader, scheduler, optimizer, BATCH_SIZE, N_EPOCHS, criterion):
     
     loss_vector_training = []

     for epoch in range(N_EPOCHS):
            
            loss_sum = 0

            scheduler.step()
    
            for i, (time_series, labels) in enumerate(train_loader):

                model.zero_grad()

                predictions = model(time_series)

                loss = criterion(predictions, labels)

                loss.backward()

                optimizer.step()
               
                loss_sum += loss.item()
        
            # divide loss_sum by the number of objects in a batch (total number of objects in training set / N_BATCHES)

            loss_total = loss_sum / BATCH_SIZE
            loss_vector_training.append(loss_total)

            print('Loss for each Epoch: ', 'Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, N_EPOCHS, loss_total))

            count = 0
            aucTotal = 0
            AUC_vector = []
 
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
            AUC_vector.append(auc)

            print('Epoch [{}/{}], Validation AUC: {}'.format(epoch+1, N_EPOCHS, auc))       

     return model, optimizer 
