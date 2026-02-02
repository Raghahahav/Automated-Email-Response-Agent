# Email Assistant with RAG and Streamlit

This repository contains a Retrieval-Augmented Generation (RAG)-based email assistant built using LangChain, Groq LLM, and Streamlit. The assistant is designed to help users draft professional and compliant email replies based on a provided knowledge base.

## Features

- **RAG-based Email Assistant**: Combines the power of LangChain and Groq LLM to generate context-aware email replies.
- **Streamlit Frontend**: A user-friendly interface for pasting client emails and generating replies.
- **Knowledge Base Integration**: Uses FAISS and HuggingFace embeddings for efficient retrieval of relevant information.
- **Customizable Design**: Aesthetic and modern frontend design with customizable themes.

## Project Structure

```
Email/
├── backend/
│   ├── __init__.py
│   └── email_agent.py
├── data/
│   └── faiss_index/
│       └── index.faiss
├── frontend/
│   ├── __init__.py
│   └── app.py
├── config.py
├── ingest.py
├── main.py
├── rag_chain.py
├── README.md
└── requirements.txt
```

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd Email
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Build the Knowledge Base**:
   Run the following command to ingest the knowledge base and build the FAISS index:

   ```bash
   python ingest.py
   ```

5. **Run the Application**:
   Start the Streamlit app:

   ```bash
   PYTHONPATH=$(pwd) .venv/bin/streamlit run frontend/app.py
   ```

6. **Access the App**:
   Open your browser and navigate to the local URL provided by Streamlit (e.g., `http://localhost:8501`).

## Usage

1. Paste the client's email into the input box on the Streamlit app.
2. Click the "Generate Reply" button.
3. View the AI-generated reply in the output section.

## Dependencies

- Python 3.9+
- LangChain
- Groq LLM
- FAISS
- HuggingFace Transformers
- Streamlit
- Python-dotenv
- Sentence Transformers

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.
# Automated-Email-Response-Agent
