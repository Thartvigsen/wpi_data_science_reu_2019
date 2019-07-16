import torch
import torch.nn as nn
import numpy as np

def gruDCell(self, x, h, m, dt, x_prime):
    x_prime = m * x + (1-m) * x_prime
    
    # -- compute value / state decays
    delta_x = torch.exp(-torch.max(self._zeros_x, self.gamma_x(dt)))
    delta_h = torch.exp(-torch.max(self._zeros_h, self.gamma_h(dt)))

    # -- apply decays --
    h = delta_h * h
    x = m * x + (1-m) * (delta_x * x_prime + (1-delta_x) * delta._x_mean)
 
    # x becomes the new estimated input post-imputation
    
    # -- gating functions
    combined = torch.cat((x, h, m), dim=2)
    r = torch.sigmoid(self.r(combined))
    z = torch.sigmoid(self.z(combined))
    new_combined = torch.cat((x, torch.mul(r, h), m), dim=2)
    h_tilde = self.tanh(self.h(new_combined))
   
    return h, x_prime

def forward(self, sequence):
    self._x_mean = self._x_mean[:self._input_dim]
    h = self.initHidden()
    x_prime = torch.zeros(self._input_dim)
    for i in range(len(sequence)):
        x = sequence[i, :, :self._input_dim].unsqueeze(0)
        mask = sequence[i, :, self._input_dim:2*self._input_dim].unsqueeze(0)
        diff = sequence[i, :, 2*self._input_dim:].unsqueeze(0)
        h, x_prime = self.gruDCcell(x, h, mask, diff, x_prime)

    output = self.out(h).squeeze(0) # Remove time dimension
    predictions = self.out_nonlin(output)
    return predictions
