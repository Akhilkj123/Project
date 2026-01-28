cd sdn-rag
python3 -m venv venv
source venv/bin/activate



pip install -U \
  langchain-core \
  langchain-community \
  langchain-groq \
  chromadb \
  fastembed
