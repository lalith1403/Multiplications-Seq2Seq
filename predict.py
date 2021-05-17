import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pickle 
# from model import CustomLSTM


data = torch.load('./data/data.pt').cuda()
class CustomLSTM(nn.Module):
    def __init__(self,data):
        super(CustomLSTM, self).__init__()
        self.data = data
        self.embedding = nn.Embedding(data.shape[0] ,data.shape[1])
        self.lstm = nn.LSTM(12,3,2,batch_first=True)
        self.linear = nn.Linear(3,1)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        input_to_lstm = self.embedding(x)
        lstm_output, (_, _) = self.lstm(input_to_lstm)
        output = self.linear(lstm_output)
        # output.dtype = torch.float64
        return output

model = CustomLSTM(data).cuda()
model.load_state_dict(torch.load('model.pth'))

targets = torch.load('./data/targets.pt')
max_target = max(targets)
print(model(data)*max_target)
