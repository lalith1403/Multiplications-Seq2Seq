import torch
import torch.nn as nn
import pickle
from preprocess_dataset import load_data
from model import Vocabulary

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
        
        # Use attention_output instead of encoder_outputs for context vector calculation
        context_vector = torch.sum(attention_output, dim=1)
        
        output = self.fc(context_vector)
        return output
# Function to load the vocabulary
def load_vocab(path):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    vocab = Vocabulary(sentences)
    return vocab

# Convert numbers to words (simplified version, consider using inflect library for a robust solution)
def number_to_words(number):
    num_dict = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 0: "zero"}
    return " ".join(num_dict[int(digit)] for digit in str(number) if digit.isdigit())

# Main prediction function
def predict(model, sentence, vocab):
    model.eval()
    numericalized = torch.tensor([vocab.numericalize(sentence)], dtype=torch.long)
    with torch.no_grad():
        prediction = model(numericalized)
    return prediction.item()

# Load the vocabulary and the model
VOCABULARY_PATH = './data/vocabulary.pkl'
MULTIPLICATION_PAIRS = './data/multiplication_pairs.pkl'
MODEL_PATH = './trained_model.pth'

sentences, results = load_data(VOCABULARY_PATH, MULTIPLICATION_PAIRS)

vocab = load_vocab(VOCABULARY_PATH)
model = Seq2SeqAttention(vocab_size=len(vocab.word2idx), embed_size=256, hidden_size=512, output_size=1, num_heads=8)
model.load_state_dict(torch.load(MODEL_PATH))

# Accept two numbers from the user
num1 = input("Enter the first number: ")
num2 = input("Enter the second number: ")

# Convert numbers to words and format the input sentence
sentence = f"{number_to_words(num1)} multiplied by {number_to_words(num2)} equals"

# Predict
prediction = predict(model, sentence, vocab)

# Calculate the actual result
actual_result = int(num1) * int(num2)

# Print the predicted and actual results
print(f"Model's prediction: {prediction}")
print(f"Actual result: {actual_result}")