import torch
from torch import nn
from torchtyping import TensorType

from .normalization import LayerNorm

class ModulationOutput:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha[:,None]
        self.beta = beta[:,None]
        self.gamma = gamma[:,None]
    
    def first_step(self, x): # [b,n,d]
        return x * (1 + self.alpha) + self.beta
    
    def second_step(self, x):
        return x * self.gamma

class DoubleModBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, 6 * dim)
    
    def forward(self, cond): # [b,d]
        cond = self.act(cond)
        params = self.fc(cond)
        alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = params.chunk(6, dim = -1) # Break into 6 parts
        return [
            ModulationOutput(alpha_1, beta_1, gamma_1),
            ModulationOutput(alpha_2, beta_2, gamma_2)
        ]     

class FinalMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ln = LayerNorm(dim)

        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond): #[b,n,d],[b,d]
        cond = self.act(cond)

        x = self.ln(x)
        scale, shift = self.fc(cond).chunk(2,dim=-1) # 2*[b,d]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1) # Add n dim to both

        return x * (1. + scale) + shift