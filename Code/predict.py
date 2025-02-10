import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, matthews_corrcoef, roc_auc_score
from model import ToxiPep_Model
from dataset import convert_to_graph_channel



Pep_residue2idx = {
    '[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
    'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7,
    'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22
}


def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
    return sequences



def transform_Pep_to_index(sequences, residue2idx):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]
        token_index.append(seq_id)
    return token_index


def pad_sequence(token_list, max_len=51):
    data = []
    for seq in token_list:
        seq = [Pep_residue2idx['[CLS]']] + seq
        seq.extend([Pep_residue2idx['[PAD]']] * (max_len - len(seq)))
        data.append(seq)
    return data


class PeptideDataset(Dataset):
    def __init__(self, sequences, graph_features):
        self.sequences = sequences
        self.graph_features = graph_features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]), torch.tensor(self.graph_features[idx]))


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for input_ids, graph_features in data_loader:
            input_ids, graph_features = input_ids.to(device), graph_features.to(device)
            outputs = model(input_ids, graph_features, device)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类概率
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    return predictions, probabilities


parser = argparse.ArgumentParser(description='Peptide toxicity prediction')
parser.add_argument('-i', type=str, required=True, help='Input FASTA file')
parser.add_argument('-o', type=str, required=True, help='Output Prediction TXT file')
args = parser.parse_args()


sequences = read_fasta(args.i)


indexed_sequences = transform_Pep_to_index(sequences, Pep_residue2idx)
padded_sequences = pad_sequence(indexed_sequences)
graph_features = [convert_to_graph_channel(seq) for seq in sequences]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(Pep_residue2idx)
d_model, d_ff, n_layers, n_heads, max_len = 256, 512, 2, 4, 50
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 50,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len, structural_config=structural_config).to(device)
model.load_state_dict(torch.load("best_model.pth"))


test_dataset = PeptideDataset(padded_sequences, graph_features)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


predictions, probabilities = evaluate_model(model, test_loader, device)


idx_to_residue = {v: k for k, v in Pep_residue2idx.items()}


with open(args.o, 'w') as f:
    for seq, pred, prob in zip(indexed_sequences, predictions, probabilities):
        seq_str = ''.join([idx_to_residue[i] for i in seq if i in idx_to_residue and i not in [Pep_residue2idx['[CLS]'], Pep_residue2idx['[PAD]']]])
        adjusted_prob = prob if pred == 1 else 1 - prob
        f.write(f"{seq_str}, {pred}, {adjusted_prob:.6f}\n")

print(f"Prediction results saved as {args.o}")
