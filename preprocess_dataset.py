import re 
import random 
import torch
import pickle

class PreprocessDataset():
    """Class to Preprocess Dataset present at file_name
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.unique_tokens = ['<unk>','<pad>']
        self.target = []
        self.token_idx = {} 
        self.MAX_LENGTH = 0

    def load_data(self):
        """load data from the data file

        Returns:
            list: list of all data pairs 
        """
        data = open(self.file_name, 'r',encoding='utf8')
        data = data.readlines()
        return data
    
    def preprocess_data(self):
        """Preprocess dataset
        """
        unprocessed_data = self.load_data()
        for _, pair in enumerate(unprocessed_data):
            self.data.append(re.split('-| |', pair))
    
    def generate_targets(self):
        """get the output of multiplication
        """
        for pair in self.data:
            target = pair.pop()
            self.target.append(int(target))


        for i in self.data:
            self.MAX_LENGTH = max(len(i), self.MAX_LENGTH)
    
    def random_data_display(self):
        """Randomly display data from the preprocessed dataset
        """
        num = random.randint(0, len(self.data))
        print(self.data[num])

    def generate_vocabulary(self):
        """generate the unique tokens
        """
        for indv_sentence in self.data:
            for indv_token in indv_sentence[:-1]:
                if indv_token not in self.unique_tokens:
                    self.unique_tokens.append(indv_token)
    
    def generate_token_idx(self):
        "generate token_idx dictionary"
        for id, token in enumerate(self.unique_tokens):
            self.token_idx[token] = id 
    
    def pad_sentences(self):
       for i in range(len(self.data)):
           length = len(self.data[i])
           self.data[i] += ["<pad>" for _ in range (self.MAX_LENGTH - length)] 

    def replace_words_with_idx(self):
        for i in range(len(self.data)):
            for j in range(self.MAX_LENGTH):
                self.data[i][j] = self.token_idx[self.data[i][j]]


FILE_NAME = './data/multiplication_pairs.txt'

preprocess_dataset = PreprocessDataset(FILE_NAME)
preprocess_dataset.load_data()
preprocess_dataset.preprocess_data()
preprocess_dataset.generate_targets()
preprocess_dataset.generate_vocabulary()
preprocess_dataset.generate_token_idx()
preprocess_dataset.pad_sentences()
preprocess_dataset.replace_words_with_idx()
data = preprocess_dataset.data

data = torch.LongTensor(data)
torch.save(data,'./data/data.pt')
targets = preprocess_dataset.target
targets = torch.LongTensor(targets)
torch.save(targets,'./data/targets.pt')
print(preprocess_dataset.token_idx)

with open("./data/token_idx.pkl",'wb') as handle:
    pickle.dump(preprocess_dataset.token_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
