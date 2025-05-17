

import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from atom_feature import convert_to_graph_channel

# 残基到索引的映射
Pep_residue2idx = {
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

def transform_Pep_to_index(sequences, residue2idx):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]
        token_index.append(seq_id)
    return token_index

def pad_sequence(token_list, max_len=51):
    data = []
    for tokens in token_list:
        seq = [Pep_residue2idx['[CLS]']] + tokens
        n_pad = max_len - len(seq)
        seq.extend([Pep_residue2idx['[PAD]']] * max(n_pad, 0))
        data.append(seq[:max_len])
    return data

def load_data_from_fasta(file_path, max_len=51):

    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        current_seq = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):

                header = line[1:]

                label = 1 if header.lower().startswith('pos') else 0
                labels.append(label)
                current_seq = ''
                sequences.append(current_seq)
            else:

                sequences[-1] += line


    indexed_sequences = transform_Pep_to_index(sequences, Pep_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences, max_len=max_len)


    graph_features = [convert_to_graph_channel(seq) for seq in sequences]

    return padded_sequences, graph_features, labels


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, graph_features, labels):
        self.input_ids = input_ids
        self.graph_features = graph_features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.graph_features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
