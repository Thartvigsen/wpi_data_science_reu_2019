'''
This class assigns the parameter of the model. This code is what to edit to change these parameters
'''

import torch
import torch.nn as nn

class params():
    
    def __init__(self):
        super(params, self).__init__()
        
        self.HIDDEN_DIMENSION = 128
        self.N_LAYERS = 2
        self.BATCH_SIZE = 64
        self.DROPOUT = 0.5
        self.N_EPOCHS = 250
        self.LEARNING_RATE = 0.005
        self.N_FEATURES = 112
        self.N_CLASSES = 20

        self.criterion = nn.functional.binary_cross_entropy
       
        self.optimizer = 0
        self.scheduler = 0

    def setOptimizer(self, model):

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .99)
