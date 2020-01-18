<!-- ![NeuralMatching Logo](readme/logo.jpg) -->
# NeuralMatching: An Open-source Neural Question Answering Toolkit

## Motivation
Rewrite the pairwise-qa
## Models
- [Attentive Pooling Networks](https://arxiv.org/abs/1602.03609)

- [Inner Attention based Recurrent Neural Networks for Answer Selection](http://www.aclweb.org/anthology/P16-1122)

## Requirements
- python3

- Tensorflow = 1.8

## File Tree

````
.
|
|- /
|  |- README.md ______________________________ # -
|  |- config.py ______________________________ # - parameters of the code
|  |- data_helper.py _________________________ # - data propressing
|  |- evaluation.py __________________________ # -
|  |- package.json ___________________________ # -
|  |- qa.code-workspace ______________________ # -
|  |- run.py _________________________________ # -
|
|  |- models/
|    |- __init__.py __________________________ # -
|    |- blocks.py ____________________________ # -
|    |- model.py _____________________________ # - basis model 
|    |- rnn_model.py _________________________ # - implemented by model
|
|  |- data/ question answering dataset
|
|    |- wiki/
|      |- dev.txt ____________________________ # -
|
|      |- test/
|      |- test.txt ___________________________ # -
|
|      |- train/
|      |- train.txt __________________________ # -
|
|    |- trec/
|      |- dev.txt ____________________________ # -
|
|      |- test/
|      |- test.txt ___________________________ # -
|
|      |- train/
|      |- train-all.txt ______________________ # -
|      |- train.txt __________________________ # -
|      |- train_orin.txt _____________________ # -


````

## Run the Code

````
python run.py
````
## Contributor

-   [@ZhanSu](https://github.com/shuishen112)
