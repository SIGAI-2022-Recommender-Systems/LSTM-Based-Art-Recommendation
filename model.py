import torch
import torch.nn as nn

class recSysNet(nn.Module):
    def __init__(self, device, output_size, embedding_dim, hidden_size, num_layers, dropout, bidirectional):
        super(recSysNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size #hidden state
        self.bidirectional_multiplier = 2 if bidirectional else 1
        self.device = device
        
        #self.embed = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first = True, 
                          dropout = dropout if num_layers>1 else 0, 
                          bidirectional = bidirectional) #lstm
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*self.bidirectional_multiplier, output_size) #fully connected last layer
        
    
    def forward(self,x):
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers*self.bidirectional_multiplier, 
                                                  x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers*self.bidirectional_multiplier, 
                                                  x.size(0), self.hidden_size)).to(self.device) #internal state
        # Propagate input through LSTM
        #x = self.embed(x)
        x, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        x = self.dropout(x)
        x = self.fc(x)
        return x
