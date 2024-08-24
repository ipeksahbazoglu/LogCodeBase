import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_f1_score
from collections import Counter


class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, dataset):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1)
        h2 = global_add_pool(h2)
        h3 = global_add_pool(h3) 

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h

def train(model, optimizer, criterion, data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      pred = out.argmax(dim = 1)
      
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
      
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss, train_acc

def test(model, optimizer, criterion, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc, test_loss

def validation(model, optimizer, criterion, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
      val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
      val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
      return val_acc, val_loss

def train_model(dataset, filepath, verbose = False):

      data = dataset.data
      model = GIN(64, dataset)
      criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
      optimizer = torch.optim.Adam(model.parameters(), lr=0.010, weight_decay=0.0100)
      for epoch in range(1, 201):
            loss, train_acc = train(model, optimizer, criterion, data)
            test_acc, loss_test = test(model, optimizer, criterion, data)
            val_acc, val_loss = validation(model, optimizer, criterion, data)
            if verbose & (epoch % 10 == 0):
                  print(f'Epoch: {epoch:03d}, Train_Acc: {train_acc:.4f}, Test_Acc: {test_acc:.4f}, Val_Acc: {val_acc:.4f}')

      torch.save(model.state_dict(), filepath)
      return model

def load_model(filepath, dataset):
      model = GIN(64, dataset)
      model.load_state_dict(torch.load(filepath))
      model.eval()

      return model

def generate_loss(filepath, dataset, data_p):
      criterion = torch.nn.CrossEntropyLoss()
      model = load_model(filepath, dataset)
      model.eval()

      out = model(data_p.x, data_p.edge_index)
      t_loss = criterion(out[data_p.train_mask], data_p.y[data_p.train_mask])

      return t_loss.item()


def generate_output(model, dataset, data_p):
      

      data = dataset.data
      h_p = model(data_p.x, data_p.edge_index)

      out_p = h_p.argmax(dim = 1)
      
      fscore = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask],
                           num_classes=7, average = None).flatten().numpy()
      
      f1_macro_p = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask], 
                          num_classes=7, average = 'macro').item()
      
      correct_nodes = (out_p[data.test_mask] == data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()
      mis_nodes = (out_p[data.test_mask] != data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()

      TP = [x[1] for x in sorted(Counter((data.y[data.test_mask])[correct_nodes].flatten().numpy()).items())]
      FN = [x[1] for x in sorted(Counter((data.y[data.test_mask])[mis_nodes].flatten().numpy()).items())]

      FP = [x[1] for x in sorted(Counter(out_p[data.test_mask][mis_nodes].flatten().numpy()).items())]
      


      return list(fscore) + [f1_macro_p], TP, FN, FP

