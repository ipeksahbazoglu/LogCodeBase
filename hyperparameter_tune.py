import optuna
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

import pickle 

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Define the Optuna objective function
def objective(trial, dataset):
    data = dataset[0]
    
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])  # Categorical choices
    dropout = trial.suggest_categorical('dropout', [0.5, 0.6, 0.7, 0.8, 0.9])  # Categorical choices
    lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2, 1e-1])  # Categorical choices
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3, 1e-2, 1e-1])  # Categorical choices


    model = GCN(dataset, hidden_channels=hidden_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience = 10
    counter = 0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on the validation set
        if epoch % 10 == 0:
            model.eval()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_labels = data.y[data.val_mask].cpu()
            val_preds = pred[data.val_mask].cpu()
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    return best_val_f1

def get_best_params(dataset, filepath, verbose = True):
    # Run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset), n_trials=20)
    
    if verbose: 
        print("Best parameters: ", study.best_params)
        print("Best F1 Score: ", study.best_value)

        # Save the best parameters to a file
    with open(filepath, 'wb') as f:
        pickle.dump(study.best_params, f)

def load_params(filepath):
# Load the best parameters from the JSON file
    with open(filepath, 'rb') as f:
        config = pickle.load(f)
    return config
