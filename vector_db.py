from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load the document
loader = TextLoader("documents.txt", encoding="utf-8")
documents = loader.load()

# Step 2: Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

# Step 3: Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store the chunks and embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="mini_rag"
)

print("Documents successfully stored in ChromaDB!")

# Step 5: Ask a question
question = input("\nEnter your question: ")

# Step 6: Retrieve relevant chunks
results = vectorstore.similarity_search(
    question,
    k=2
)

# Step 7: Display the results
print("\nRelevant information:")

for result in results:
    print(result.page_content)