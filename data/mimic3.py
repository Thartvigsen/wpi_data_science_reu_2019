
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import visdom

# In[ ]:


class SimpleSignal(torch.utils.data.Dataset):
    def __init__(self):
        super(SimpleSignal, self).__init__()
        self.data, self.labels = self.generate_data()
        self.train_ix, self.test_ix, self.validation_ix = self.get_ix_splits()
    
    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]
    
    def __len__(self):
        return len(self.data)

    def get_ix_splits(self):
        split_props = [0.8, 0.1, 0.1] # Train/validation/test split proportions
        indices = range(len(self.data))
        split_points = [int(len(self.data)*i) for i in split_props]
        train_ix = np.random.choice(indices,
                                    split_points[0],
                                    replace=False)
        test_ix = np.random.choice((list(set(indices)-set(train_ix))),
                                    split_points[1],
                                    replace=False)
        validation_ix = np.random.choice((list(set(indices)-set(test_ix))), split_points[2], replace=False)

        return train_ix, test_ix, validation_ix

    def load_data(self, file_name):
        return open(file_name, "r+")

    def generate_data(self):
        data = torch.load("time_series.pt")
        masks = torch.load("masks.pt")
        
        for i in range(6261):
            for j in range(59):
                patient = np.asarray(data[i,...,j])
                patientMasks = np.asarray(masks[i,...,j])
                
                realPatient = []
                for k in range(192):
                    if(patientMasks[k]==0):
                        realPatient.append(k)
                
                if(len(realPatient)==0):
                   data[i, ..., j] = torch.from_numpy(np.zeros(192)) 
                else:

                    mean = np.mean(realPatient)
                    std = np.std(realPatient)
                    upperBound = mean+2*std
                    lowerBound = mean-2*std

                    offset = 0.0
                    counter = 0.0

                    for k in range(len(realPatient)):
                        if(realPatient[k] > upperBound or realPatient[k] < lowerBound):
                            realPatient[k] = 55555.55
                            offset += 55555.55
                        else:
                            counter+=1                        
                
                    mean = (np.sum(realPatient)-offset)/counter
                    for k in range(192):
                        if(patientMasks[k]==1):
                            realPatient.insert(k, mean)
                        if(realPatient[k]==55555.55):
                            realPatient[k] = mean
                
                    data[i,...,j] = torch.from_numpy(np.asarray(realPatient))         

        labels = torch.load("labels.pt")
        return data, labels



# In[ ]:


HIDDEN_DIMENSION = 100
N_LAYERS = 5 
BATCH_SIZE = 10
N_EPOCHS = 10
N_EPOCHS_2 = 50
N_EPOCHS_3 = 100
LEARNING_RATE = 0.1
N_FEATURES = 59
N_CLASSES = 20

drop = 0.5
epochs_drop = 1.0

def step_decay(epoch, lrate):

    lrate = lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate

# --- create the dataset ---
data = SimpleSignal()

# --- define the data loaders ---
train_sampler = SubsetRandomSampler(data.train_ix) # Random sampler for training indices
test_sampler = SubsetRandomSampler(data.test_ix) # Random sampler for testing indices
validation_sampler = SubsetRandomSampler(data.validation_ix)

train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=BATCH_SIZE, 
                                           sampler=train_sampler,
                                           shuffle=False, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=BATCH_SIZE, 
                                          sampler=test_sampler,
                                          shuffle=False, drop_last=True)

validation_loader = torch.utils.data.DataLoader(dataset=data,batch_size=BATCH_SIZE,sampler=validation_sampler, shuffle=False, drop_last=True)


# --- define your model here ---
class RNN(nn.Module):
    def __init__(self, HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS, BATCH_SIZE):
        super(RNN, self).__init__()
        self.N_LAYERS = N_LAYERS
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_DIMENSION = HIDDEN_DIMENSION
        self.N_CLASSES = N_CLASSES
        # --- define mappings here ---
    
        self.LSTM = torch.nn.LSTM(N_FEATURES, HIDDEN_DIMENSION, N_LAYERS)
        self.out = torch.nn.Linear(HIDDEN_DIMENSION, N_CLASSES)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, X):
        # --- define forward pass here ---



        state = (torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIMENSION), torch.zeros(self.N_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIMENSION))
        
        X = torch.transpose(X, 0, 1)

        hidden, state = self.LSTM(X, state)
        output = self.out(hidden[-1])
        prediction = self.softmax(output)
        return prediction
        
HIDDEN_DIMENSION_2 = 5
# --- initialize the model and the optimizer ---
model = RNN(HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS, BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Using the Adam optimizer - don't worry about the details, it's going to update the network's weights.
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#vis = visdom.Visdom()

#loss_window = vis.line(X=np.column_stack(torch.zeros((1)).cpu()), Y=np.column_stack(torch.zeros((1)).cpu()), opts=dict(xlabel='Epoch',ylabel='Loss', title = 'Training Loss', legend=["Number of LSTMS = 1"]))

LEARNING_RATE_2 = 0.1
HIDDEN_DIMENSION_3 = 10
HIDDEN_DIMENSION_4 = 40

N_LAYERS_2 = 3
N_LAYERS_3 = 5
N_LAYERS_4 = 10

LEARNING_RATE_3 = 0.00001

model2 = RNN(HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS_2, BATCH_SIZE)
optimizer_2 = torch.optim.Adam(model2.parameters(), lr = LEARNING_RATE)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=1, gamma=0.1)

model3 = RNN(HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS_3, BATCH_SIZE)
optimizer_3 = torch.optim.Adam(model3.parameters(), lr = LEARNING_RATE)
scheduler_3 = torch.optim.lr_scheduler.StepLR(optimizer_3, step_size=1, gamma=0.1)

#model4 = RNN(HIDDEN_DIMENSION, N_CLASSES, N_FEATURES, N_LAYERS_4, BATCH_SIZE)
#optimizer_4 = torch.optim.Adam(model4.parameters(), lr = LEARNING_RATE)



# --- training the model ---
for epoch in range(N_EPOCHS_2):
    loss_sum = 0
    loss_sum_2 = 0
    loss_sum_3 = 0
    loss_sum_4 = 0
    ultimate_correct = 0
    ultimate_correct_2 = 0
    ultimate_correct_3 = 0
    ultimate_correct_4 = 0
    total = 0
    total_2 = 0
    total_3 = 0
    total_4 = 0
    scheduler.step()
    scheduler_2.step()
    scheduler_3.step()

    for i, (time_series, labels) in enumerate(train_loader): # Iterate through the training batches
        # --- Forward pass ---
        model.zero_grad()
        model2.zero_grad()
        model3.zero_grad()
        #model4.zero_grad()

        predictions = model(time_series)
        predictions_2 = model2(time_series)
        predictions_3 = model3(time_series)
        #predictions_4 = model4(time_series)

        total += 10
        total_2 += 10
        total_3 += 10
        total_4 += 10

        for i in range(10):
            if(predictions[i][1]<=.5):
                if(labels[i]==0):
                    ultimate_correct += 1
            elif(labels[i]==1):
                ultimate_correct += 1

            if(predictions_2[i][1]<=.5):
                if(labels[i]==0):
                    ultimate_correct_2 += 1
            elif(labels[i]==1):
                ultimate_correct_2 += 1

            if(predictions_3[i][1]<=.5):
                if(labels[i]==0):
                    ultimate_correct_3 += 1
            elif(labels[i]==1):
                ultimate_correct_3 += 1
 
            #if(predictions_4[i][1]<=.5):
            #   if(labels[i]==0):
            #      ultimate_correct_4 += 1
            #elif(labels[i]==1):
            #    ultimate_correct_4 += 1

        # --- Compute gradients and update weights ---
        loss = criterion(predictions, labels)
        loss_2 = criterion(predictions_2, labels)
        loss_3 = criterion(predictions_3, labels)
        #loss_4 = criterion(predictions_4, labels)
        loss_sum += loss.item()
        loss_sum_2 += loss_2.item()
        loss_sum_3 += loss_3.item()
        #loss_sum_4 += loss_4.item()

        loss.backward()
        loss_2.backward()
        loss_3.backward()
        #loss_4.backward()
        optimizer.step()
        optimizer_2.step()
        optimizer_3.step()
        #optimizer_4.step()


    vloss_sum = 0
    vloss_sum_2 = 0
    vloss_sum_3 = 0
    vloss_sum_4 = 0
    vultimate_correct = 0
    vultimate_correct_2 = 0
    vultimate_correct_3 = 0
    vultimate_correct_4 = 0
    vtotal = 0
    vtotal_2 = 0
    vtotal_3 = 0
    vtotal_4 = 0
    for i, (time_series, labels) in enumerate(validation_loader): # Iterate through the training batches
        # --- Forward pass ---
        model.zero_grad()
        model2.zero_grad()
        model3.zero_grad()
        #model4.zero_grad()

        predictions = model(time_series)
        predictions_2 = model2(time_series)
        predictions_3 = model3(time_series)
        #predictions_4 = model4(time_series)

        vtotal += 10
        vtotal_2 += 10
        vtotal_3 += 10
        vtotal_4 += 10

        for i in range(10):
            if(predictions[i][1]<=.5):
                if(labels[i]==0):
                    vultimate_correct += 1
            elif(labels[i]==1):
               vultimate_correct += 1

            if(predictions_2[i][1]<=.5):
                if(labels[i]==0):
                    vultimate_correct_2 += 1
            elif(labels[i]==1):
                vultimate_correct_2 += 1

            if(predictions_3[i][1]<=.5):
                if(labels[i]==0):
                    vultimate_correct_3 += 1
            elif(labels[i]==1):
                vultimate_correct_3 += 1
 
            #if(predictions_4[i][1]<=.5):
            #   if(labels[i]==0):
            #      ultimate_correct_4 += 1
            #elif(labels[i]==1):
            #    ultimate_correct_4 += 1

        # --- Compute gradients and update weights ---
        vloss = criterion(predictions, labels)
        vloss_2 = criterion(predictions_2, labels)
        vloss_3 = criterion(predictions_3, labels)
        #loss_4 = criterion(predictions_4, labels)
        vloss_sum += vloss.item()
        vloss_sum_2 += vloss_2.item()
        vloss_sum_3 += vloss_3.item()
        #loss_sum_4 += loss_4.item()


 #   vis.line(X=torch.ones((1,1)).cpu() * epoch, Y=torch.Tensor([loss_sum/80]).unsqueeze(0).cpu(), win=loss_window, name="Number of LSTMS = 1", update='append')
  #  vis.line(X=torch.ones((1,1)).cpu() * epoch, Y=torch.Tensor([loss_sum_2/80]).unsqueeze(0).cpu(), win=loss_window, name="Number of LSTMS = 3", update='append')
   # vis.line(X=torch.ones((1,1)).cpu() * epoch, Y=torch.Tensor([loss_sum_3/80]).unsqueeze(0).cpu(), win=loss_window, name="Number of LSTMS = 5", update='append')
   # vis.line(X=torch.ones((1,1)).cpu() * epoch, Y=torch.Tensor([loss_sum_4/80]).unsqueeze(0).cpu(), win=loss_window, name="Number of LSTMS = 10", update='append')
    print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {}%'.format(epoch+1, N_EPOCHS_2, vloss_sum/vtotal, vultimate_correct/vtotal))

total = 0
total2 = 0
total3 = 0
total4 = 0
correct = 0
correct2 = 0
correct3 = 0
correct4 = 0

for i, (time_series, labels) in enumerate(test_loader):

    predictions = model(time_series)
    predictions2 = model2(time_series)
    predictions3 = model3(time_series)
    #predictions4 = model4(time_series)

    total += 10
    total2 += 10
    total3 += 10
    total4 += 10

    for i in range(10):
        if(predictions[i][1]<=.5):
            if(labels[i]==0):
                correct += 1
        elif(labels[i]==1):
            correct += 1

        
        if(predictions2[i][1]<=.5):
            if(labels[i]==0):
                correct2 += 1
        elif(labels[i]==1):
            correct2 += 1
        
        if(predictions3[i][1]<=.5):
            if(labels[i]==0):
                correct3 += 1
        elif(labels[i]==1):
            correct3 += 1

        #if(predictions4[i][1]<=.5):
            #if(labels[i]==0):
               # correct4 += 1
        #elif(labels[i]==1):
            #correct4 += 1

y = [correct/total, correct2/total, correct3/total]

#accuracy_window = vis.bar(y, opts=dict(xlabel='Number of LSTMS',ylabel='accuracy', title = 'Testing Accuracy', legend=["Number of LSTMS = 1", "Number of LSTMS = 3", "Number of LSTMS = 5"]))

print('Accuracy: {}%, {}/{}'.format((correct/total), correct, total))
print('Accuracy: {}%, {}/{}'.format((correct2/total), correct2, total))
print('Accuracy: {}%, {}/{}'.format((correct3/total), correct3, total))
