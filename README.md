# Transformer

This Transformer model is based on the original ["Attention is all you need" paper](https://arxiv.org/pdf/1706.03762) and trained on the[tinyShakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The token size is 1 because we just used characters. 

´bigram.py´ Contains a simple Bigrammodel, which uses an embedding space of (vocab_size, vocab_size). In this type of model we count occurences of `token_a` after `token_b`. This means that the generation of the next token only depends on the previous token.

`transformer.py` is the fully fledged decoder of the original *Attention is ll you need* paper. It contains the token embedding layer with 384 dimensions, a positional encoding layer for each of the blocks. The blocks containing, one multi-head attion, one feed forward and two normalization layers. After that block there is one more linear transformation layer that creates the logits that are later used in combination with a softmax to generate the next character.