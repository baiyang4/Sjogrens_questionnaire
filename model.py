import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstm, self).__init__()

        self.out_layer = nn.Sequential(
            nn.LSTM(input_size, hidden_size, batch_first=True),
        )
        self.out_layer_1 = nn.Sequential(
            nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True),
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, 2, dropout=0.1, batch_first=True)
        # self.layer_norm = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

        # self.lrelu = nn.LeakyReLU()
        self.out = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        y_lstm, _ = self.out_layer(x)
        y_lstm = self.norm(y_lstm)
        # y_1, _ = self.out_layer_1(y_lstm)
        # y_1 = self.norm_1(y_1)
        y = self.dense(y_lstm)
        # out = self.out(y[:,-1,:])
        return y[:,-1,:], 0, 0



class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size, dropout=0.2):
        super(SelfAttention, self).__init__()

        self.m = input_size
        self.attention_size = attention_size
        self.dropout = dropout

        self.K = nn.Linear(in_features=self.m, out_features=self.attention_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.attention_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.attention_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=self.m, bias=False),
            # nn.Tanh(),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        K = torch.t(self.K(x).squeeze(0))  # ENC (n x m) => (n x H) H= hidden size
        Q = torch.t(self.Q(x).squeeze(0))  # ENC (n x m) => (n x H) H= hidden size
        V = torch.t(self.V(x).squeeze(0))

        logits = torch.div(torch.matmul(K.transpose(1, 0), Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=-1)
        # weight = F.sigmoid(logits)
        mid_step = torch.matmul(V, weight.t())
        # mid_step = torch.matmul(V, weight)
        attention = torch.t(mid_step).unsqueeze(0)

        attention = self.output_layer(attention)

        return attention, weight


class SelfAttentionLstm(nn.Module):
    def __init__(self, input_size, hidden_size, model_type):
        super(SelfAttentionLstm, self).__init__()

        self.model_type = model_type
        self.out_layer = nn.Sequential(
            nn.LSTM(input_size, hidden_size, batch_first=True),
        )
        self.out_layer_1 = nn.Sequential(
            nn.LSTM(hidden_size, hidden_size, batch_first=True),
        )
        self.out_layer_2 = nn.Sequential(
            nn.LSTM(hidden_size, hidden_size, batch_first=True),
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, 2, dropout=0.1, batch_first=True)
        # self.layer_norm = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, 1)
        self.sensor_att = SensorLstm(input_size)
        # self.lrelu = nn.LeakyReLU()
        self.out = nn.Sigmoid()
        self.att_layer = SelfAttention(input_size=hidden_size, attention_size=hidden_size)

    def forward(self, x):
        # print(x.shape)
        # x, beta = self.sensor_att(x)
        y_lstm, _ = self.out_layer(x)
        y_lstm = self.norm(y_lstm)
        if self.model_type < 9:
            y_lstm, _ = self.out_layer_1(y_lstm)
            y_lstm = self.norm_1(y_lstm)
        # y_lstm, _ = self.out_layer_2(y_lstm)
        # y_lstm = self.norm_2(y_lstm)
        attention1, weight = self.att_layer(y_lstm)
        # out1 = self.norm_2(attention1 + y_1)
        y = self.dense(attention1)
        # out = self.out(y[:,-1,:])
        return y[:,-1,:], weight[-1], weight[:,-1]


