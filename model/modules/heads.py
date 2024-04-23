import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AveragePooler(nn.Module):
    def __init__(self, hidden_size=185):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states.permute(0, 2, 1)  # [bz, 768, 185]
        pooled_output = torch.squeeze(self.dense(first_token_tensor), 2)  # [bz, 768, 1]
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
