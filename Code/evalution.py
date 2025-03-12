from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, classification_report
import pandas as pd
from model import *
from dataset import *


from model import *
from dataset import *
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
import pandas as pd


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)

            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy, all_preds, all_labels


def calculate_metrics(all_labels, all_preds):
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return acc, sen, spe, mcc, auc


def save_metrics_to_csv(metrics, filename="metrics.csv"):
    df = pd.DataFrame([metrics], columns=["ACC", "Sensitivity", "Specificity", "MCC", "AUC"])
    df.to_csv(filename, index=False)
    print(f"The results of the evaluation have been saved to {filename}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(dna_residue2idx)
d_model, d_ff, n_layers, n_heads, max_len = 256, 512, 2, 4, 50
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 50,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}


test_sequences, test_graph_features, test_labels = load_data_from_csv(
    'test.csv')

test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)


model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                           structural_config=structural_config).to(device)

model.load_state_dict(torch.load("best_model.pth"))
criterion = nn.CrossEntropyLoss()


_, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)


acc, sen, spe, mcc, auc = calculate_metrics(all_labels, all_preds)
metrics = {"ACC": acc, "Sensitivity": sen, "Specificity": spe, "MCC": mcc, "AUC": auc}

print(f"ACC: {acc:.4f}, Sensitivity: {sen:.4f}, Specificity: {spe:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
save_metrics_to_csv(metrics)



