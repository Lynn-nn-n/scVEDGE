import torch
from typing import Iterable
from mlp import MLP
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int = 1,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.2,
        use_spectral_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.discriminator = MLP(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_spectral_norm = use_spectral_norm,
            **kwargs,
        )
        self.linear_out = spectral_norm(torch.nn.Linear(n_hidden, 1))
        #self.linear_out = torch.nn.Linear(n_hidden, 1)
        self.regularize = torch.nn.Sigmoid()
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        f_D = self.discriminator(x, *cat_list)
        p_real = self.regularize(self.linear_out(f_D))
        return f_D, p_real  

class Critic(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int = 1,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.discriminator = MLP(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.fc3 = spectral_norm(nn.Linear(n_hidden, 1))  
    
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        f_D = self.discriminator(x, *cat_list)
        p_real = self.fc3(f_D)
        return f_D, p_real

    def gradient_penalty(self, real_data, fake_data, lambda_gp=10):

        epsilon = torch.rand(real_data.size(0), 1).to(real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

       
        interpolated_score = self.forward(interpolated)

       
        gradients = torch.autograd.grad(
            outputs=interpolated_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_score).to(real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

     
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return lambda_gp * penalty
