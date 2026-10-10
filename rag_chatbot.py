from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load documents
loader = TextLoader("documents.txt", encoding="utf-8")
documents = loader.load()

# Step 2: Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="mini_rag_chatbot"
)

# Step 5: Connect to the LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)
print("MiniRAG Chatbot is ready!")
print("Type 'exit' to stop.")

# Step 6: Ask questions
while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    # Combine retrieved text
    context = "\n\n".join(
        document.page_content for document in results
    )

    # Prepare the prompt
    prompt = f"""
    Answer the question using the context below.

    If the answer is not present in the context,
    say that the information is not available in the documents.

    Context:
    {context}

    Question:
    {question}
    """

    # Generate the answer
    response = llm.invoke(prompt)

    print("\nAI:", response.content)