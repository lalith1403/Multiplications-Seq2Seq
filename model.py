import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pickle 

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
        lstm_output = self.dropout(lstm_output)
        output = self.linear(lstm_output)
        # output.dtype = torch.float64
        return output

data = torch.load('./data/data.pt').cuda()
model = CustomLSTM(data).cuda()
# model.load_state_dict(torch.load('model.pth'))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

y_pred = model(data).cuda()
targets = torch.load('./data/targets.pt')
targets = torch.reshape(targets, (targets.shape[0],1))

targets = torch.tensor(targets,dtype = torch.float).cuda()
targets = targets/ max(targets)

lambda1 = lambda epoch: epoch // 300
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
# print(y_pred[:,1,:].shape, targets.shape)

for epoch in range(20000):
    model.zero_grad()
    y_pred = model(data)
    y_pred = y_pred[:,-1,:]
    loss = loss_function(targets, y_pred)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch%100 ==0:
        print(epoch+1,": ", loss.data)
torch.save(model.state_dict(), 'model.pth')