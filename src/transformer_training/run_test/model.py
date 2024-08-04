import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]  # Adjusted for batch size handling

class TransformerForSlideClassification(nn.Module):
    def __init__(self, fv_extractor_size=1024, d_model=256, nhead=2, num_encoder_layers=2, dim_feedforward=512, dropout=0.1, max_seq_length=15000):
        super(TransformerForSlideClassification, self).__init__()
        self.fc = nn.Linear(fv_extractor_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, src):
        src = self.fc(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        out = self.classifier(memory.mean(dim=1))
        return out
