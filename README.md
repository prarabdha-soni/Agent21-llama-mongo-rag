Commands:-

git clone https:xxxxxxxxx

pip install torch transformers sentence-transformers flask annoy numpy


python mongo-llm.py



🚀 MongoRAG: AI-Powered MongoDB Query Generator
MongoRAG is an intelligent system that combines Retrieval-Augmented Generation (RAG) and Generative AI (Llama 2) to generate optimized MongoDB queries from natural language inputs.

✨ Features
🔍 1) RAG-Based Query Retrieval
Stores sample MongoDB queries in a FAISS vector database.
Uses sentence embeddings to find similar queries.
If a query is found with a similarity score < 0.5, it is retrieved instantly, reducing latency.

🤖 2) Generative AI Query Creation (Llama 2)
If no similar query is found, Llama 2 generates a new query.
The model is prompted to return strict JSON output to ensure structured queries.