__all__ = ['FTMLP_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
import math
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
# Cell

class FTMLP_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 n_layers:int=3, d_model=128, dropout:float=0., fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        

        # Backbone
        self.backbone = FTblock(c_in=c_in,context_window=context_window, n_layers=n_layers, d_model=d_model, dropout=dropout)

        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual




        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, target_window,context_window, head_dropout=head_dropout)
        
    
    def forward(self, z):
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
        else:
            z = z.permute(0, 2, 1)

        z = self.backbone(z)
        # model
        z = self.head(z)
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, target_window,context_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        self.linear = nn.Linear(context_window,target_window)
        self.dropout = nn.Dropout(head_dropout)

            
    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FTblock(nn.Module):
    def __init__(self, c_in,context_window,n_layers=3, d_model=128, dropout=0.):
        
        
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.encoder = Block(c_in,context_window,dropout=dropout, n_layers=n_layers)

    def forward(self, x) -> Tensor:


        x = x.permute(0,2,1)
        z = self.encoder(x)

        
        return z
# Cell
class Block(nn.Module):
    def __init__(self,c_in,context_window, dropout=0., n_layers=1):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(c_in,context_window,dropout) for i in range(n_layers)])
    def forward(self, src:Tensor):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class Filter(nn.Module):
    def __init__(self, feature_num, sequence_len):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(sequence_len, 2, dtype=torch.float32))
        self.sequence_len = sequence_len
    def forward(self, x):

        weight = torch.view_as_complex(self.complex_weight)
        x = torch.fft.fft (x,norm='ortho',dim = -2)
        x = x * weight
        x = torch.fft.ifft(x,norm='ortho',dim = -2)
        x = x.real
        return x
class Temporalmodule(nn.Module):
    def __init__(self, d_model,d_model1, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False,dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model1)
        self.bn1 = nn.BatchNorm1d(d_model1)
        self.bn2 = nn.BatchNorm1d(d_model1)
        self.bn3 = nn.BatchNorm1d(d_model1)
        self.skip_connect = skip_connect
        hidden_features = int(d_model * mlp_ratio)
        self.act = act_layer()
        self.T_fc1 = nn.Linear(d_model,hidden_features)
        self.T_fc2 = nn.Linear(hidden_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.filter = Filter(d_model1,d_model)
    def forward(self, x):
        xs = self.bn(x)
        xs = self.filter(xs)
        xs = self.bn3(xs)
        xs = self.T_fc1(xs)
        xs = self.act(xs)
        xs = self.dropout(xs)
        xs = self.bn1(xs)
        xs = self.T_fc2(xs)
        xs = self.dropout(xs)
        xs = self.bn2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Featuremodule(nn.Module):
    def __init__(self, d_model,d_model1, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False,dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.skip_connect = skip_connect
        hidden_features = int(d_model * mlp_ratio)
        self.act = act_layer()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, seq_length = x.size()
        squeezed = self.squeeze(x).view(batch_size, num_channels)
        excitation = self.excitation(squeezed)
        excitation = excitation.view(batch_size, num_channels, 1)
        xs = x * excitation
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
class EncoderLayer(nn.Module):
    def __init__(self,c_in,context_window, dropout=0.):
        super().__init__()
        self.featuremodule = Featuremodule(c_in,context_window,skip_connect=False,dropout=dropout)
        self.temporalmodule = Temporalmodule(context_window,c_in,skip_connect=False,dropout=dropout)

        self.dropout_attn = nn.Dropout(dropout)

        self.dropout_ffn = nn.Dropout(dropout)
    def forward(self, src:Tensor) -> Tensor:

        x = self.featuremodule(src)
        x = self.temporalmodule(x)
        x = x + src
        return x
