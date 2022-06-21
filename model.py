import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re
from collections import Counter
from preprocess_dataset import load_data
# Assuming preprocessing has been done and we have `sentences` and `results`

class Vocabulary:
    def __init__(self, sentences):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.build_vocab(sentences)
    
    def build_vocab(self, sentences):
        word_counts = Counter(word for sentence in sentences for word in sentence.split())
        for word, _ in word_counts.items():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def numericalize(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.split()]

class MultiplicationDataset(Dataset):
    def __init__(self, sentences, results, vocab, max_length=None):
        self.sentences = [vocab.numericalize(sentence) for sentence in sentences]
        self.results = results
        self.vocab = vocab
        self.vocab_size = len(vocab.word2idx)
        self.max_length = max_length if max_length else max(len(s) for s in self.sentences)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Pad sentences to the maximum length
        padded_sentence = sentence + [self.vocab.word2idx["<pad>"]] * (self.max_length - len(sentence))
        return torch.tensor(padded_sentence, dtype=torch.long), torch.tensor(self.results[idx], dtype=torch.float)

# Model
# class Seq2SeqAttention(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, output_size):
#         super(Seq2SeqAttention, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
#         self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.attention = nn.Linear(hidden_size * 2, 1)
    
#     def forward(self, x):
#         embedded = self.embedding(x)
#         encoder_outputs, (hidden, cell) = self.encoder(embedded)
#         # Decoder with attention
#         decoder_output, _ = self.decoder(embedded, (hidden, cell))
#         attention_weights = torch.softmax(self.attention(torch.cat((encoder_outputs, decoder_output), dim=2)), dim=1)
#         context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        
#         output = self.fc(context_vector)
#         return output

class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_heads):
        super(Seq2SeqAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # Initialize multi-head self-attention
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # Apply self-attention to the encoder outputs
        attention_output, _ = self.self_attention(encoder_outputs, encoder_outputs, encoder_outputs)
        
        # Decoder with attention output instead of original encoder output
        decoder_output, _ = self.decoder(embedded, (hidden, cell))

        # Apply self-attention to decoder outputs
        attention_output, _ = self.self_attention(decoder_output, decoder_output, decoder_output)
        
        # Use attention_output instead of encoder_outputs for context vector calculation
        context_vector = torch.sum(attention_output, dim=1)
        
        output = self.fc(context_vector)
        return output


# Assuming we have a function to load data and split it
MULTIPLICATION_PAIRS = "./data/multiplication_pairs.pkl"
VOCAB = "./data/vocabulary.pkl"

sentences, results = load_data(VOCAB, MULTIPLICATION_PAIRS)
vocab = Vocabulary(sentences)
dataset = MultiplicationDataset(sentences, results, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Seq2SeqAttention(vocab_size=len(vocab.word2idx), embed_size=256, hidden_size=512, output_size=1, num_heads=8)

# model = Seq2SeqAttention(vocab_size=len(vocab.word2idx), embed_size=256, hidden_size=512, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
from tqdm import tqdm

if __name__ == "__main__":
    for epoch in range(10):
        for inputs, targets in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation and Prediction (simplified for demonstration)
    def predict(model, sentence, vocab):
        model.eval()
        numericalized = torch.tensor([vocab.numericalize(sentence)], dtype=torch.long)
        with torch.no_grad():
            prediction = model(numericalized)
        return prediction.item()

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

    # Example usage
    sentence = "three multiplied by four equals"
    prediction = predict(model, sentence, vocab)
    print(f"Predicted result: {prediction}")