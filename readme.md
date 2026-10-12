# 🤖 MiniRAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangChain, Hugging Face, ChromaDB, Ollama, and Streamlit.

MiniRAG retrieves relevant information from documents and uses a locally running Large Language Model (LLM) to generate context-aware answers through an interactive web interface.

## 🚀 Features

- Document loading using LangChain
- Text chunking with overlapping sections
- Text embeddings using Hugging Face Sentence Transformers
- Vector storage using ChromaDB
- Semantic similarity search
- Context-based answer generation using a local LLM
- Interactive web interface using Streamlit
- Local AI inference using Ollama and Llama 3.2

## 🛠️ Technologies Used

- Python
- LangChain
- Streamlit
- Hugging Face
- ChromaDB
- Ollama
- Llama 3.2 (3B)

## 📂 Project Structure

```text
miniRag/
├── app.py
├── rag_chatbot.py
├── chunking.py
├── embeddings_demo.py
├── vector_db.py
├── documents.txt
├── README.md
└── .gitignore
```

## ⚙️ How It Works

1. Load documents from a text file.
2. Split the documents into smaller text chunks.
3. Convert the chunks into vector embeddings.
4. Store and search the embeddings using ChromaDB.
5. Retrieve relevant chunks based on the user's question.
6. Send the retrieved context to the local LLM.
7. Generate and display the answer in the Streamlit interface.

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd miniRag
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Install the required Python packages for this project.

If you have a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

### 5. Download the Ollama Model

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running.

### 6. Start the Web Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

### Alternative: Run the Command-Line Chatbot

```bash
python rag_chatbot.py
```

## 🎯 Learning Outcomes

- Understanding Retrieval-Augmented Generation
- Working with text embeddings and vector databases
- Implementing semantic similarity search
- Integrating a local Large Language Model
- Building an interactive AI application

## 🔮 Future Improvements

- PDF and TXT document uploads
- Displaying retrieved document sources
- Chat history and conversational memory
- Improved retrieval accuracy
- RAG evaluation
- Cloud deployment

## 👨‍💻 Author

MiniRAG — A hands-on project exploring RAG, vector databases, and local LLM applications.