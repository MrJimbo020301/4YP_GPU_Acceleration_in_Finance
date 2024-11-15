import torch 
import torch.nn as nn 
import torch.nn.functional as F # this has activation functions 

# option 1 (create nn classes)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        nn.Sigmoid
        nn.Tanh
        nn.LeakyReLU
        nn.Softmax
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        torch.softmax
        torch.tanh 
        return out

# option 2 (use activation functions directly in forward pass)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        F.leaky_relud
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out