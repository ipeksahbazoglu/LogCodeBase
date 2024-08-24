import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch_geometric


from torcheval.metrics.functional import multiclass_f1_score
from collections import Counter


class GCN(torch.nn.Module):
        
    def __init__(self, dataset, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, config['hidden_channels'])
        self.conv2 = GCNConv(config['hidden_channels'], dataset.num_classes)
        self.dropout = config['dropout']
        self.hidden_channels = config['hidden_channels']
    
    def forward(self, x, edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = F.dropout(x, p= self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

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

def train_model(dataset, config, filepath, verbose = False):

      data = dataset.data
      model = GCN(dataset, config)
      criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
      optimizer = torch.optim.Adam(model.parameters(), lr= config['lr'], weight_decay= config['weight_decay'])
      for epoch in range(1, 201):
            loss, train_acc = train(model, optimizer, criterion, data)
            test_acc, loss_test = test(model, optimizer, criterion, data)
            val_acc, val_loss = validation(model, optimizer, criterion, data)
            if verbose & (epoch % 10 == 0):
                  print(f'Epoch: {epoch:03d}, Train_Acc: {train_acc:.4f}, Test_Acc: {test_acc:.4f}, Val_Acc: {val_acc:.4f}')
      config["accuracy"]  = {'Train_Acc': f"{train_acc:.4f}", "Test_Acc": f"{test_acc:.4f}", "Val_Acc": f"{val_acc:.4f}"} 
      config['state_dict'] = model.state_dict()
      torch.save(config, filepath)
      return model



def load_model(filepath: str, dataset:torch_geometric.datasets.planetoid.Planetoid):
    """ initiate GCN model with data, use pre-trained state values from specified file path.

    Args:
        filepath (str): _description_
        dataset (torch_geometric.datasets.planetoid.Planetoid): _description_
    """
    config = torch.load(filepath)
    model = GCN(dataset, config)
    model.load_state_dict(config['state_dict'])
    model.eval();
    
    return model


def generate_output(model, dataset, data_p):
      

      data = dataset.data
      h_p = model(data_p.x, data_p.edge_index)

      out_p = h_p.argmax(dim = 1)
      
      fscore = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask],
                           num_classes=dataset.num_classes, average = None).flatten().numpy()
      
      f1_macro_p = multiclass_f1_score(out_p[data.test_mask], data.y[data.test_mask], 
                          num_classes= dataset.num_classes, average = 'macro').item()
      
      correct_nodes = (out_p[data.test_mask] == data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()
      mis_nodes = (out_p[data.test_mask] != data.y[data.test_mask]).nonzero().squeeze().flatten().numpy()

      TP = [x[1] for x in sorted(Counter((data.y[data.test_mask])[correct_nodes].flatten().numpy()).items())]
      FN = [x[1] for x in sorted(Counter((data.y[data.test_mask])[mis_nodes].flatten().numpy()).items())]

      FP = [x[1] for x in sorted(Counter(out_p[data.test_mask][mis_nodes].flatten().numpy()).items())]
      


      return list(fscore) + [f1_macro_p], TP, FN, FP




