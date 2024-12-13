## Text data preprocessing pipeline (Reference: https://huggingface.co/docs/tokenizers/en/index)
1. Normalization
   1. Unicode Normalization
   2. StripAccents
2. Pretokenization example (white-space)
3. Model example (BPE/Unigram/WordLevel/WordPiece)
4. Post processing: additional transformation like adding additional special tokens


## Special Tokens
Special tokens are additional tokens that are added to the vocabulary so they can be tokenized into ids. Common special tokens are
1. [UNK] unknown token 
2. [EOS] end of sentence
3. [BOS] beginning of sentence
4. [PAD] padding


# Tokenization is at the heart of LLM 
Many of the issues are due to tokenization issues


## Types of tokenizers
1.  Byte-Pair Encoding (BPE)
    1.  BPE can handle any unknown word as it breaks down unknown words into subwords and indivdual characters
2.  WordPiece
3.  Unigram
4.  SentencePiece - uses 'space' in the set of characters as some languages like chinese, japanese do not have spaces
    1. Uses BPE encoding / Unigram / char / word


## BPE
Find the most common byte pair and replace with a placeholder and keeps repeating this process 
1. UTF-8 encoding is a rule of interpreting the character bytes
   1. For the first 128 "bytes" eg \x00 to \x80, they are single characters which corresponds to the ascii values
   2. If the the byte is more than \x80, it signals a multi byte sequence with the length
2. The special tokens that are added are not part of the BPE algorithm

## Sentencepieces
sentence pieces runs BPE on the unicode code points directly
1. Rare codepoints is either mapped into UNK token or if byte_fallback is turned on, it encodes them with utf-8 and then encodes the raw bytes



## Tiktokenizer webapp
https://tiktokenizer.vercel.app/



## Tokenization library
1. Huggingface tokenizer https://github.com/huggingface/tokenizers
   1. Documentation: https://huggingface.co/docs/tokenizers/index
2. Google SentencePiece https://github.com/google/sentencepiece
3. OpenAI BPE https://github.com/openai/tiktoken


## Tokenization tutorial
1. AndrewAndrej Karpathy tokenizer from scratch: https://www.youtube.com/watch?v=zduSFxRajkE&t=2098s