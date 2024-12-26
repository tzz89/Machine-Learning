
# Overview
Retreival and Ranking system can be use for retriving relevant information and also skills like querying DB/ Coding / Function calling


## Embedding compression
1. Matryoshka Representations (MRL) 
   1. The embedding model is trained in such a way that the first X embeddings holds more information that subsequent embeddings
   2. The users can truncate the embeddings to fit their use case


## Models that support MRL (Text only)
1. mixedbread AI: https://huggingface.co/mixedbread-ai
2. Nomic AI: https://huggingface.co/collections/nomic-ai/nomic-embed-65c0426827a5fdca81a87b89
3. openAI: https://openai.com/index/new-embedding-models-and-api-updates/
4. JINA AI: https://huggingface.co/jinaai


## Embeddings/Reranker Models
1. https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d


## Vector DB
1. Faiss
2. Milvus
3. Chroma
4. Qdrant
5.  Weaviate
   


## Retreival Systems
| **Metric**                  | **Description**                                                                                   | **Pros**                                                                 | **Limitations**                                                          | **Common Use Cases**                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Precision**               | Fraction of retrieved documents that are relevant.                                                | Simple and intuitive; focuses on relevance of results.                  | Ignores non-retrieved relevant documents.                                | Search engines, information retrieval where precision is critical.    |
| **Recall**                  | Fraction of relevant documents that are retrieved.                                                | Measures completeness of retrieval.                                      | Ignores irrelevant retrieved documents; may overemphasize coverage.      | Legal document retrieval, healthcare research.                        |
| **F1 Score**                | Harmonic mean of precision and recall.                                                            | Balances precision and recall for a single measure.                     | Requires both precision and recall; less interpretable alone.            | Applications needing balance between precision and recall.            |
| **Average Precision (AP)**  | Average of precision values at different recall levels for a query.                               | Captures overall retrieval quality and ranking order.                   | Computationally expensive; depends on relevance scores.                  | Evaluation of retrieval models for multi-query tasks.                 |
| **Mean Average Precision (MAP)** | Mean of AP across multiple queries.                                                          | Aggregates performance across queries; reflects ranking quality.        | Assumes binary relevance; computationally intensive.                     | Benchmarking retrieval systems, search engines.                       |
| **Reciprocal Rank (RR)**     | Reciprocal of the rank of the first relevant document retrieved.                                  | Simple, emphasizes finding at least one relevant document quickly.      | Focuses only on the first relevant result.                               | First-response retrieval systems, QA systems.                         |
| **Mean Reciprocal Rank (MRR)** | Average RR over multiple queries.                                                             | Aggregates performance across queries; emphasizes early relevance.      | Sensitive to outliers where no relevant document is retrieved.           | FAQs, chatbot answer retrieval.                                        |
| **Normalized Discounted Cumulative Gain (nDCG)** | Measures ranking quality using graded relevance, prioritizing higher-ranked documents.    | Considers relevance levels and rank positions.                          | Requires relevance grading; complex to compute.                          | Personalized search, multimedia retrieval.                            |
| **R-Precision**             | Precision when the number of retrieved documents equals the number of relevant documents.         | Balances retrieved and relevant counts.                                 | Hard to compute for very large datasets.                                 | Dataset-specific retrieval evaluation.                                |
| **Hit Rate**                | Proportion of queries with at least one relevant document retrieved.                              | Simple to compute; good for "at least one match" scenarios.             | Does not consider ranking or multiple relevant results.                  | Quick search tools, content discovery.                                |
| **Coverage**                | Fraction of the total relevant documents retrieved across all queries.                            | Useful for understanding system completeness.                           | Ignores rank and relevance of retrieved documents.                       | Large-scale retrieval systems, diversity-focused retrieval.           |
| **Latent Semantic Indexing (LSI)** | Evaluates the quality of semantic retrieval using latent features.                         | Captures latent relationships in data.                                  | Computationally expensive; requires tuning.                              | Semantic search, document similarity tasks.                           |
| **Query Latency**           | Time taken to retrieve results for a query.                                                       | Essential for real-time applications.                                   | Does not measure quality or relevance of results.                        | Real-time retrieval systems, search engines.                          |
| **User Satisfaction**       | Measures user-perceived quality of retrieved results (e.g., via surveys).                         | Directly reflects user experience.                                      | Subjective and hard to standardize.                                      | User-centric systems like e-commerce search.                          |


## Reranker
Uses the query and the retreived embeddings (cross encoder) to create a score. Performance of the ranking systems can be evaluated like the retreival system like recall@10 and Precision@10


| **Metric**                  | **Description**                                                                                   | **Pros**                                                                 | **Limitations**                                                          | **Common Use Cases**                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Precision@K**             | Fraction of relevant items in the top K results.                                                 | Easy to interpret, focuses on relevance in top results.                 | Ignores results beyond K; does not consider the total number of relevant items. | Search engines, recommendation systems where precision is critical.   |
| **Recall@K**                | Fraction of all relevant items retrieved in the top K results.                                    | Focuses on completeness, useful for datasets with many relevant items.  | Ignores irrelevant results in the top K; less informative for small K.  | Information retrieval, recall-critical applications like healthcare.  |
| **F1 Score**                | Harmonic mean of Precision@K and Recall@K.                                                       | Balances precision and recall.                                          | Requires both precision and recall to compute; less interpretable alone. | Applications needing balance between precision and recall.            |
| **Mean Reciprocal Rank (MRR)** | Average reciprocal rank of the first relevant result for multiple queries.                       | Evaluates ranking quality of the first relevant result.                 | Sensitive to the position of the first relevant result only.             | Search engines, question-answering systems.                           |
| **Normalized Discounted Cumulative Gain (nDCG)** | Measures ranking quality using graded relevance, emphasizing higher-ranked items.             | Considers relevance levels; discounts lower positions logarithmically.  | Requires relevance scores; complex to compute.                           | Personalized recommendations, web search.                             |
| **Mean Average Precision (MAP)** | Average precision across all queries, taking ranking order into account.                      | Reflects both precision and ranking order.                              | Computationally expensive; requires relevance judgments for all queries. | Evaluation of ranking algorithms across large datasets.               |
| **Hit Rate@K**              | Proportion of queries where at least one relevant item appears in the top K results.              | Simple to compute, good for "at least one relevant result" scenarios.   | Does not consider ranking quality or multiple relevant items.            | Content discovery, item recommendations.                              |
| **Coverage**                | Fraction of total catalog items shown in recommendations across all users.                       | Evaluates diversity and item exposure.                                  | May not account for relevance or user satisfaction.                      | Recommendation systems needing diverse item exposure.                 |
| **Serendipity**             | Measures how unexpected but useful recommendations are.                                           | Encourages innovative suggestions.                                      | Hard to quantify; subjective interpretation.                             | Novel item recommendations, entertainment platforms.                  |
| **Diversity**               | Measures variety of recommended items across categories or genres.                                | Ensures users see diverse recommendations.                              | May conflict with relevance.                                             | E-commerce, multimedia streaming platforms.                           |
| **Latency**                 | Time taken to generate and display recommendations or ranked results.                             | Important for user experience.                                          | Does not directly measure relevance or ranking quality.                  | Real-time recommendation systems, search engines.                     |
| **Click-Through Rate (CTR)** | Proportion of recommended items clicked by users.                                                | Reflects user engagement.                                               | Influenced by presentation and user interface.                           | Ad recommendations, user behavior analysis.                           |
| **Conversion Rate**         | Proportion of recommendations leading to a specific action (e.g., purchase).                      | Directly ties recommendations to business goals.                        | Does not measure recommendation quality explicitly.                      | E-commerce, targeted advertising.                                     |



## References
1. Jina Clip V2: https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/
2. Jina Reranker:https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/
3. Matryoshka Representation Learning and Adaptive Semantic Search: https://www.youtube.com/watch?v=IbfdvzPwZTg