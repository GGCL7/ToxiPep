import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from model import *
from dataset import *

import torch.nn.functional as F

from model import *
from dataset import *

import torch.nn.functional as F


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:

        input_ids, graph_features, labels = batch
        input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, graph_features, device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    return total_loss / len(train_loader)



def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:

            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy



vocab_size = len(dna_residue2idx)
d_model = 256
d_ff = 512
n_layers = 2
n_heads = 4
max_len = 50


structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 50,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

train_sequences, train_graph_features, train_labels = load_data_from_csv(
    'train.csv')
test_sequences, test_graph_features, test_labels = load_data_from_csv(
    'test.csv')


train_dataset = MyDataSet(train_sequences, train_graph_features, train_labels)
test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)


train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                           structural_config=structural_config).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

n_epochs = 100
best_accuracy = 0.0
best_model_path = "best_model.pth"

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'新最佳模型保存于 Epoch {epoch + 1}，准确度: {best_accuracy:.4f}')

print(f'训练完成。最高准确度: {best_accuracy:.4f}，模型已保存到 {best_model_path}')


