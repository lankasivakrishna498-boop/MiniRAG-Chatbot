from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example text
text = "RAG retrieves information from documents."

# Convert text into an embedding
embedding = model.encode(text)

# Display the embedding
print("Text:", text)
print("Embedding:", embedding)
print("Embedding length:", len(embedding))