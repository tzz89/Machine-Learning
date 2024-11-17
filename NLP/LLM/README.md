## How to utilize LLM
1. API from openai, antropic etc
2. Local/private server
   1. https://github.com/ollama/ollama
   2. https://github.com/ggerganov/llama.cpp
   3. https://github.com/nomic-ai/gpt4all
   4. https://github.com/Mozilla-Ocho/llamafile
   5. https://github.com/janhq/jan
   6. https://github.com/Lightning-AI/litgpt


## Stages of training LLM
1. Building LLM model
   1. Data preparation and sampling
2. Pretraining Foundation model (Usually only done in large enterprises)
3. FineTuning for personal assistant (Instruction) / Classifer etc
   1. Preference tuning can help boost Instruction fine-tuning (Generate Multiple responses and choose 1 to me better)
4. Continue pretraining: Adding latest knowledge

## Creating dataset for LLM training
The dataset for pretraining LLM is done via data sampling with sliding window
![Alt text](../assets/DataSamplingSlidingWindow.jpg)

## Evaluation of LLM
1. MMLU: measuring massive multitask language understanding
2. AlpacaEval: measure against GPT4
3. LMSYS chatbot arena: Crowd sourcing eval


## Tutorial
1. Overview of LLM development: https://www.youtube.com/watch?v=kPGTx4wcm_w

## Reference
1. Build LLM from scratch book: https://github.com/rasbt/LLMs-from-scratch


## Opensource LLMs / Datasets
1. List of opensource LLM and datasets https://github.com/eugeneyan/open-llms