from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load the document
loader = TextLoader("documents.txt", encoding="utf-8")
documents = loader.load()

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# Split the document
chunks = splitter.split_documents(documents)

# Display the chunks
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)