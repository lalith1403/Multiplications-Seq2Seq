# Presenting Multiplication as a Sequence-to-Sequence Problem

Neural Networks are inherently bad at performing mathematical operations. For example, getting the neural networks to multiply two numbers will most certainly result in an erroneous result. In this project, an attention based seq2seq model is used to model the problem of multiplication. 


### Approach

The dataset is created by generating 10000 pairs of random integers between 0 and 99. The pairs are then converted to strings, and the digits of the numbers are converted to their textual representation. The pairs are then saved to a pickle file.

The model is a seq2seq model with an attention mechanism. The model is trained on the dataset for 10 epochs. The model is then evaluated on the same dataset using the mean absolute error.

### Results

The problem of multiplication modelled as a sequence to sequence problem demonstrates the incapability of neural networks to perform mathematical operations.

The model is able to learn to multiply the numbers, but the model is not able to learn to multiply the numbers in a way that is nearly accurate.

### Code

#### Dataset Generation

```bash
python generate_dataset.py
```

#### Training the model

```bash
python model.py
```

#### Evaluating the model

```bash
python predict.py
```
