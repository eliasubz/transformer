# Transformer
### Overview
This Transformer model is based on the original "Attention is all you need" [paper](https://arxiv.org/pdf/1706.03762) and trained on the tinyShakespeare [dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). In this case we only implemented the decoder because we are not interested in translation tasks.  
In the decoder block **causal masked attention** is used; hiding the trailing tokens during training. This is done to preserve the auto-regressive property of the model, meaning the output is only dependent on the tokens seen so far.  
The **token size** is 1, so each token is one character. We used a **dropout** of 20\% to decrease training time and generalize the model to prevent overfitting (regularzitation). It works by choosing 20\% of the neurons arbitrarily and setting their activations to 0 and scaling the activation of the other neurons by 1 / (1 - 0.2) = 1 / (4/5) = 5/4 = 1.25, so that the sum of remaining activations stays the same. 

### Training Setup

To train the transformer model and generate a text its important to **adjust the hyperparameters** in the beginning of `tranformer.py`. With Cuda you can expect good results after 15 minutes of training with the current settings. We also include two default hyperparameter templates for a Google Colab connected to a Nvidia T4 GPU or a laptop.

1. Download the tinyShakesPeare dataset by uncommenting the wget line `wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` or manually over this [link](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
2. (Un)comment your preferred hyperparameters and adjust to your specs in the top of `tranformer.py` (read above).
3. Run `python transformer.py`
 
### Files
`bigram.py` Contains a simple Bigrammodel, which uses an embedding space of `(vocab_size, vocab_size)` and is the baseline model for the transformer. In this type of model we count occurrences of `token_a` after `token_b`. This means that the generation of the next token only depends on the previous token.

`transformer.py` is the fully fledged decoder of the original *Attention is all you need* paper. 
- Embedding Layer: Maps tokens into a 384-dimensional space.

- Positional Encoding: Added to embeddings to provide sequence order information.

- Decoder Blocks: Each block contains multi-head self-attention, a feed-forward network, and layer normalization. 

- Output Layer: A final linear projection produces logits over the vocabulary(all possible tokens/characters), followed by softmax sampling for character generation.


