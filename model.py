import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# 定义GRU网络
class GRU(nn.Module):
    def __init__(self, input_dim, input_size, hidden_size, hidden_num_layers):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = hidden_num_layers

        # 嵌入
        self.to_patch_embedding = self.to_patch_embedding = nn.Sequential(
            Rearrange('b 1 (n d) -> b 1 n d', n=input_size),
            nn.Linear(input_dim // input_size, input_dim // input_size)
        )
        # 定义LSTM
        self.gru = nn.GRU(input_dim // input_size, hidden_size, hidden_num_layers, batch_first=True)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(hidden_size * input_size, 10)
        )

    def forward(self, x):
        batch_size = x.shape[0] # 获取批次大小
        x = rearrange(x, 'b l -> b 1 l')
        x = self.to_patch_embedding(x)
        x = x.squeeze(1)
        h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        x, h_0 = self.gru(x, h_0)
        x = rearrange(x, 'b s l -> b (s l)')
        x = self.reg(x)
        return x

