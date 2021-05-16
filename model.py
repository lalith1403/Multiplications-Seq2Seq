import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle 

class CustomLSTM(nn.Module):
    def __init__(self,data):
        super(CustomLSTM, self).__init__()
        self.data = data
        self.embedding = nn.Embedding(data.shape[0],data.shape[1])
        self.lstm = nn.LSTM(12,3,1,batch_first=True)
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        input_to_lstm = self.embedding(x)
        lstm_output, (_, _) = self.lstm(input_to_lstm)
        output = self.linear(lstm_output)
        return output



data = torch.load('data.pt')
# print(data.shape)
# data = torch.LongTensor(data)
# CustomLSTM(data)
model = CustomLSTM(data)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

y_pred = model(data)
targets = torch.load('targets.pt')
targets = torch.reshape(targets, (targets.shape[0],1))

print(y_pred[:,1,:].shape, targets.shape)
print()
for epoch in range(200):
    model.zero_grad()

    y_pred[:,-1,:]
    loss = loss_function(y_pred, targets)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pth')
    