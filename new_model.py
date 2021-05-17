import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle 

class CustomLSTM(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        # print(x.shape[0],x.shape[1])
        input_to_lstm = self.embedding(x)
        print(input_to_lstm.shape)
        lstm_output, (last_hidden_state, last_cell_state) = self.lstm(input_to_lstm)
        return input_to_lstm



data = torch.load('data.pt')
with open('token_idx.pkl', 'rb') as handle:
    token_idx = pickle.load(handle)

EMBEDDING_DIM = 12
HIDDEN_DIM = 6

model = CustomLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(token_idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

with torch.no_grad():
    inputs = data[0]
    print(model(inputs))