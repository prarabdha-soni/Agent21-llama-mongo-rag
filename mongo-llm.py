from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)

# Load Llama 2 Model and Tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model

print("Loading Llama 2 model... (This may take a few minutes)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Load in FP16 for memory efficiency
    device_map="auto"  # Auto-detect GPU or CPU
)

# Load embedding model for similarity search
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Sample MongoDB queries for RAG (these should be stored in a real DB)
query_examples = [
    {"text": "Find all users older than 30", "query": '{"$match": {"age": {"$gt": 30}}}'},
    {"text": "Get all employees from the sales department", "query": '{"$match": {"department": "Sales"}}'},
    {"text": "Find orders placed in the last 7 days",
     "query": '{"$match": {"order_date": {"$gte": ISODate("2024-03-20")}}}'}
]

# Create FAISS Index for fast retrieval
dimension = 384  # Embedding size of MiniLM
index = faiss.IndexFlatL2(dimension)

# Store embeddings
query_texts = [q["text"] for q in query_examples]
query_embeddings = embedding_model.encode(query_texts)
query_embeddings = np.array(query_embeddings).astype("float32")
index.add(query_embeddings)

# Map indices to queries
query_map = {i: q["query"] for i, q in enumerate(query_examples)}


def retrieve_similar_queries(user_request, top_k=1):
    """
    Retrieve the most relevant MongoDB query using FAISS.
    """
    query_embedding = embedding_model.encode([user_request]).astype("float32")
    D, I = index.search(query_embedding, top_k)

    if D[0][0] < 0.5:  # If similarity is high (lower distance)
        return query_map.get(I[0][0], None)
    return None


def generate_mongo_query(user_request):
    """
    Converts English input into a MongoDB query using Llama 2.
    """
    retrieved_query = retrieve_similar_queries(user_request)

    if retrieved_query:
        print(f"🔍 Retrieved query: {retrieved_query}")
        return retrieved_query

    prompt = f"Convert this request to a MongoDB query and return ONLY the JSON output without explanation:\n{user_request}\n\nMongoDB Query:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=300)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract JSON output
    query_start = response.find("{")
    query_end = response.rfind("}")

    if query_start != -1 and query_end != -1:
        response = response[query_start:query_end + 1]

    return response


@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint to convert natural language input to MongoDB queries.
    """

    user_request = request.json.get("query")

    if not user_request:
        return jsonify({"error": "No query provided"}), 400

    print(f"📥 Received request: {user_request}")  # Log request

    mongo_query = generate_mongo_query(user_request)

    print(f"📤 Generated Query: {mongo_query}")  # Log response

    return jsonify({"mongo_query": mongo_query})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
