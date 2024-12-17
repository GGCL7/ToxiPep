import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=51):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.bi_gru = nn.GRU(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

    def forward(self, input_ids):
        x = self.src_emb(input_ids)
        embeddings = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        gru_output, _ = self.bi_gru(embeddings)
        return gru_output



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class Structural(nn.Module):
    def __init__(self, embedding_dim=21, max_seq_len=50, filter_num=64, filter_sizes=None):

        super(Structural, self).__init__()
        if filter_sizes is None:
            filter_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num

        self.convs = nn.ModuleList(
            [nn.Conv2d(embedding_dim, filter_num, fsz, stride=1, padding=(fsz[0] // 2, fsz[1] // 2)) for fsz in filter_sizes]
        )

        self.fc = nn.Linear(len(filter_sizes) * filter_num, 1024)
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, device):

        graph = graph.to(device)
        graph = graph.transpose(2, 3)
        graph = graph.transpose(1, 2)
        conv_outs = [F.relu(conv(graph)) for conv in self.convs]
        pooled_outs = [F.adaptive_avg_pool2d(conv_out, (1, 1)).view(graph.size(0), -1) for conv_out in conv_outs]
        concat_out = torch.cat(pooled_outs, 1)
        representation = self.fc(concat_out)
        representation = self.dropout(representation)
        return representation


class peptide(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len=50):
        super(peptide, self).__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, d_model)
        )

    def forward(self, input_ids):
        emb_out = self.emb(input_ids)
        trans_out = self.transformer_blocks(emb_out)
        pooled_output = self.pool(trans_out.transpose(1, 2)).squeeze(-1)
        logits = self.fc(pooled_output)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        attn_output_1, _ = self.multihead_attn(q, k, v)
        q = self.norm1(q + attn_output_1)

        attn_output_2, _ = self.multihead_attn(k, q, v)
        k = self.norm2(k + attn_output_2)

        return q, k



class ToxiPep_Model(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len, structural_config, cross_attention_dim=256):
        super(ToxiPep_Model, self).__init__()
        self.peptide_model = peptide(vocab_size, d_model, d_ff, n_layers, n_heads, max_len)
        self.structural_model = Structural(**structural_config)
        self.structural_linear = nn.Linear(1024, cross_attention_dim)
        self.cross_attention = CrossAttention(embed_dim=cross_attention_dim, n_heads=n_heads)
        self.fc = nn.Linear(2 * cross_attention_dim, 2)

    def forward(self, input_ids, graph_features, device):
        peptide_output = self.peptide_model(input_ids)
        structural_output = self.structural_model(graph_features, device)
        structural_output = self.structural_linear(structural_output)
        peptide_output = peptide_output.unsqueeze(1)
        structural_output = structural_output.unsqueeze(1)
        cross_peptide, cross_structural = self.cross_attention(peptide_output, structural_output, structural_output)

        cross_peptide = cross_peptide.squeeze(1)
        cross_structural = cross_structural.squeeze(1)

        combined_features = torch.cat((cross_peptide, cross_structural), dim=1)

        logits = self.fc(combined_features)
        return logits
