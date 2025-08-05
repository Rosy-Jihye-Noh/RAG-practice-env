# Income Tax Chatbot

**A Streamlit-based AI chatbot for answering questions about Korean income tax law.**  
This project provides quick and accurate answers to user questions about income tax, leveraging up-to-date legal documents and example Q&As.

---

## Features

- **Streamlit Web UI**: Intuitive, chat-style web interface
- **LangChain + OpenAI**: Advanced natural language understanding and response generation
- **Pinecone Vector Database**: Efficient document embedding and retrieval (RAG)
- **Session-based Conversation History**: Maintains context for more relevant answers
- **Real Tax Law Example Answers**: Includes sample Q&As from Korean tax law

---

## Project Structure

```
tax_env/
│
├── app.py                # Main Streamlit app (chatbot UI)
├── llm.py                # Core logic: LLM, RAG, vector DB, prompt chains
├── config.py             # Example Q&As and configuration
├── requirements.txt      # Required Python packages
├── 01_tax_rag.ipynb      # Experimental/analysis notebook
├── tax.docx              # Reference document (e.g., tax law)
├── myenv/                # (Virtual environment, if used)
└── %USERPROFILE%.pyenv/  # pyenv for Windows (Python version management, not directly related to chatbot)
```

---

## Getting Started

1. **Install Python 3.8+**  
   (Using pyenv-win or similar is recommended.)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**  
   Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=...
   PINECONE_API_KEY=...
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## Key Technologies

- **Streamlit**: Web UI framework
- **LangChain**: LLM chains, prompts, document retrieval
- **OpenAI GPT**: Natural language processing and answer generation
- **Pinecone**: Vector database for RAG
- **dotenv**: Environment variable management

---

## Example Questions

- How is income classified under Korean tax law?
- What is the tax period for income tax?
- When can I receive a withholding tax certificate?

(See `config.py` for more sample Q&As.)

---

## Notes

- The chatbot uses documents like `tax.docx` for vector-based retrieval and context-aware answers.
- The `pyenv-win` folder is for Python version management and is not directly related to the chatbot functionality.

---

## License

MIT License

---
