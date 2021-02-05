import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        
        return x

class NetworkReLU(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        act1 = self.relu(self.fc1(x))
        d = self.dropout(out1)
        act2 = self.rely(self.fc2(d))
        d = self.dropout(out2)
        act3 = self.relu(self.fc3(d))
        d = self.dropout(out3)
        out = self.fc4(x)
        
        return out