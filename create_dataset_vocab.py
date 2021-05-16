from num2words import num2words
import pickle
import random
import numpy as np
import itertools

class GenerateDataset():
    def __init__(self, vocab_file_name, data_file_name, start_num, end_num, step):
        self.vocab_file_name = vocab_file_name
        self.start_num = start_num
        self.end_num = end_num
        self.data_file_name = data_file_name
        self.step = step

    def create_dictionary(self):
        """
        Create a dictionary of all words between the given range
        """
        local_dict = {}
        
        for i in range(self.start_num, self.end_num, self.step):
            local_dict[num2words(i)] = i
        
        with open(self.vocab_file_name, 'wb') as handle:
            pickle.dump(local_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_random_number_pairs(self):
        pairs = list(itertools.combinations(range(self.start_num, int(np.sqrt(self.end_num))),2))
        return pairs
    
    def write_product_to_file(self):
        pairs = self.generate_random_number_pairs()
        for i in pairs:
            with open(self.data_file_name, 'a') as handle:
                text = str(num2words(i[0])) + " multiplied by " + str(num2words(i[1])) 
                data = i[0]*i[1]  
                handle.write(text + " " + str(data) + "\n")


VOCAB_FILENAME = "./data/vocabulary.pkl"
DATA_FILENAME = "./data/multiplication_pairs.txt"
generated_dataset = GenerateDataset(VOCAB_FILENAME, DATA_FILENAME, 0, 100000, 1)

generated_dataset.write_product_to_file()
