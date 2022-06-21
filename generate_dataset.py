import random
from num2words import num2words
import pickle

class GenerateRandomDataset():
    def __init__(self, vocab_file_name, data_file_name, start_num, end_num, num_pairs):
        self.vocab_file_name = vocab_file_name
        self.data_file_name = data_file_name.replace('.txt', '.pkl')  # Ensure we're working with a .pkl file
        self.start_num = start_num
        self.end_num = end_num
        self.num_pairs = num_pairs

    def generate_random_number_pairs(self):
        """
        Generate a specified number of random number pairs within the given range.
        """
        pairs = [(random.randint(self.start_num, self.end_num), random.randint(self.start_num, self.end_num)) for _ in range(self.num_pairs)]
        return pairs

    def write_data_to_files(self):
        """
        Generate random multiplication pairs, convert them to words, and write to .txt and .pkl files.
        """
        pairs = self.generate_random_number_pairs()
        data_list = []

        # Write sentences to a .txt file
        with open(self.data_file_name.replace('.pkl', '.txt'), 'w') as txt_file:
            for pair in pairs:
                num1, num2 = pair
                product = num1 * num2
                sentence = f"{num2words(num1)} multiplied by {num2words(num2)} equals {num2words(product)}"
                txt_file.write(sentence + '\n')
                data_list.append((sentence, product))

        # Write sentences and products to a .pkl file
        with open(self.data_file_name, 'wb') as pkl_file:
            pickle.dump(data_list, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    vocab_filename = "./data/vocabulary.pkl"
    data_filename = "./data/multiplication_pairs.txt"  # This will be replaced with .pkl in the class
    start_num = 1
    end_num = 100
    num_pairs = 10000  # Number of random pairs to generate

    dataset_generator = GenerateRandomDataset(vocab_filename, data_filename, start_num, end_num, num_pairs)
    dataset_generator.write_data_to_files()
    print("Random dataset generation complete.")