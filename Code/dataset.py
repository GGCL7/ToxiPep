import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from atom_feature import convert_to_graph_channel

dna_residue2idx = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    'A': 3,   # Alanine
    'C': 4,   # Cysteine
    'D': 5,   # Aspartic acid
    'E': 6,   # Glutamic acid
    'F': 7,   # Phenylalanine
    'G': 8,   # Glycine
    'H': 9,   # Histidine
    'I': 10,  # Isoleucine
    'K': 11,  # Lysine
    'L': 12,  # Leucine
    'M': 13,  # Methionine
    'N': 14,  # Asparagine
    'P': 15,  # Proline
    'Q': 16,  # Glutamine
    'R': 17,  # Arginine
    'S': 18,  # Serine
    'T': 19,  # Threonine
    'V': 20,  # Valine
    'W': 21,  # Tryptophan
    'Y': 22,  # Tyrosine
}





def transform_dna_to_index(sequences, residue2idx):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]
        token_index.append(seq_id)
    return token_index



def pad_sequence(token_list, max_len=51):
    data = []
    for i in range(len(token_list)):
        token_list[i] = [dna_residue2idx['[CLS]']] + token_list[i]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([dna_residue2idx['[PAD]']] * n_pad)
        data.append(token_list[i])
    return data


# 从CSV文件中读取数据
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    sequences = df['Seq'].tolist()
    labels = df['Label'].tolist()

    # 转换序列并进行填充
    indexed_sequences = transform_dna_to_index(sequences, dna_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences)

    # 提取特征矩阵
    graph_features = [convert_to_graph_channel(seq) for seq in sequences]

    return padded_sequences, graph_features, labels

# 构建PyTorch数据集
class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, graph_features, labels):
        self.input_ids = input_ids
        self.graph_features = graph_features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.graph_features[idx]), torch.tensor(self.labels[idx])
