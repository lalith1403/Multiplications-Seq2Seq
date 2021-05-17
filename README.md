# LSTM-multiplication
Neural Networks are inherently bad at performing mathematical operations. For example, getting the neural networks to multiply two numbers will most certainly result in an erroneous result. In this project, LSTMs are used to model the same. 


### Approaches

1. Pose this problem as many-to-one. The output is posed as a regression task, with input as the numbers sent in a sequence, with words represented as text. The first approach works terribly bad, given the looks of it. Probably need to find a better way to formulate the problem. 

2. Pose this problem as many-to-many. In this case both input and output are sequence. Need to code this out!