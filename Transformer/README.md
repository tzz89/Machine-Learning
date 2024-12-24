## Building Blocks for transformer 




## Postion encoding
1. Bert style:
   1. Learn embeddings at position 0 - seq lenght
2. Deberta Style
   1. Relative position encoding is fed as a seperate input to the attention
      1. the attention weight is then calculated using query_relative_position_embedding @ key_content
3. Sinosodal Decoder encoding in the original transformer is all you need paper
4. ROpe Rotary Embedding in LLama

## Regularization
1. Attention weight dropouts
2. Layer Normalization
3. RMS normalization (the idea is that scaling is more important that recentering)


## Self attention
There are 2 types of self attention
1. Self attention, the Query and Key are from same input a the Value
2. Cross Attention, the Query and key are from different input as the the Value

Multi-attention heads vs single attention heads
1. With multi attention heads, the model can learn multiple pattern and can be parallelized more efficiently
** Multi attention head have an additional final linear projection to 'mix' the output from all the different heads

Optimizations
1. There are various optimization techniques used to speed up this attention layer like "flash attention". In pytorch, we can use the scale_dot_product layer
2. K-V cache. Only the last token from the previous iteration is used in the forward method

Attention Mask
In next token prediction task, the attention weights are masked so each token can only attend to previous tokens and itself


## Pretraining task
1. Encoding Transformer
   1. Mask Language Modelling
      1. Uses Mask Tokens to force model to learn from other tokens
      2. Randomly replace Mask position with random token, or leaves token unchange to prevent overfitting to MASK token
2. Decoder Transformer
   1. Next token prediction
      1. Ouputs are offset by 1 (eg input = tokens[:-1] target = tokens[1:])