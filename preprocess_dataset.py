import pickle

def load_data(vocab_file_name, data_file_name):
    """
    Load the dataset from the pickle file and separate it into inputs and targets.

    Args:
    - vocab_file_name (str): The path to the vocabulary pickle file.
    - data_file_name (str): The path to the data pickle file containing sentences and products.

    Returns:
    - sentences (list): A list of sentences (inputs).
    - products (list): A list of products (targets).
    """
    # Load the vocabulary
    with open(vocab_file_name, 'rb') as vocab_file:
        vocabulary = pickle.load(vocab_file)
    
    # Load the data
    with open(data_file_name, 'rb') as data_file:
        data_list = pickle.load(data_file)
    
    # Separate the sentences and products
    sentences, products = zip(*data_list)  # This unzips the list of tuples into two lists
    
    return list(sentences), list(products)

if __name__ == "__main__":
    vocab_filename = "./data/vocabulary.pkl"
    data_filename = "./data/multiplication_pairs.pkl"  # Assuming the data is saved in a .pkl file
    sentences, products = load_data(vocab_filename, data_filename)
    print(f"Loaded {len(sentences)} sentences and {len(products)} products.")