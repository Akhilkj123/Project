# ==========================================================
# UNIFIED RAG FOR SDN ATTACK ANALYSIS
# Supports:
#   - Controller logs (.log)
#   - Wireshark CSVs (.csv)
# Controllers: RYU / ODL
# Focus: Threat intelligence & cross-plane reasoning
# ==========================================================

import os
import csv
import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

BASE_DATA_DIR = "/home/akhil/Desktop/Attacks/Ryu"
VECTOR_DB_DIR = "./sdn_vector_db"
COLLECTION_NAME = "sdn_attack_knowledge"
MODEL_NAME = "llama-3.1-8b-instant"   # supported Groq model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------
# SIMPLE TEXT SPLITTER (NO EXTRA DEPENDENCIES)
# ----------------------------------------------------------

def split_text(text, max_len=500):
    chunks, buf = [], ""
    for line in text.splitlines():
        if len(buf) + len(line) <= max_len:
            buf += line + "\n"
        else:
            chunks.append(buf.strip())
            buf = line + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# ----------------------------------------------------------
# LOAD CONTROLLER LOG FILES (.log)
# ----------------------------------------------------------

def load_controller_log(path):
    docs = []

    fname = os.path.basename(path).lower()
    if "dos" in fname:
        attack = "dos"
    elif "arp" in fname:
        attack = "arp_spoofing"
    elif "flow" in fname:
        attack = "flow_rule_poisoning"
    else:
        attack = "unknown"

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            text = f"""
ATTACK: {attack}
SOURCE: Controller Log
PLANE: Control
LOG: {line}
"""

            for chunk in split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "attack": attack,
                            "plane": "control",
                            "source": "controller_log"
                        }
                    )
                )

    return docs

# ----------------------------------------------------------
# LOAD WIRESHARK CSV FILES (.csv)
# ----------------------------------------------------------

def load_wireshark_csv(path):
    docs = []

    fname = os.path.basename(path).lower()
    if "arp" in fname:
        attack = "arp_spoofing"
    elif "flow" in fname:
        attack = "flow_rule_poisoning"
    elif "dos" in fname:
        attack = "dos"
    else:
        attack = "unknown"

    try:
        df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    except Exception as e:
        print(f"[!] Failed to read CSV {path}: {e}")
        return docs

    for _, row in df.iterrows():
        row_text = "\n".join([f"{k}: {row[k]}" for k in row.index])

        text = f"""
ATTACK: {attack}
SOURCE: Wireshark CSV
PLANE: Data
{row_text}
"""

        for chunk in split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "attack": attack,
                        "plane": "data",
                        "source": "pcap_csv"
                    }
                )
            )

    return docs

# ----------------------------------------------------------
# LOAD ALL DATA (LOG + CSV)
# ----------------------------------------------------------

def load_all_data(base_dir):
    documents = []

    print("[+] Loading attack artifacts")

    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)

        if file.endswith(".log"):
            print(f"[+] LOG: {full_path}")
            documents += load_controller_log(full_path)

        elif file.endswith(".csv"):
            print(f"[+] CSV: {full_path}")
            documents += load_wireshark_csv(full_path)

    return documents

# ----------------------------------------------------------
# MANUAL RAG QUERY (NO DEPRECATED CHAINS)
# ----------------------------------------------------------

def ask_rag(llm, retriever, query):
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )

    prompt = f"""
You are an SDN security researcher.

Answer the question STRICTLY using the provided context.
If evidence is insufficient, clearly state that.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return llm.invoke(prompt)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    documents = load_all_data(BASE_DATA_DIR)

    if not documents:
        print("[!] No data found.")
        return

    print(f"[+] Total chunks indexed: {len(documents)}")

    embeddings = FastEmbedEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    print("\n✅ RAG READY (Ctrl+C to exit)\n")

    while True:
        try:
            q = input(">> ").strip()
            if not q:
                continue

            ans = ask_rag(llm, retriever, q)
            print("\nANSWER:\n", ans.content)
            print("\n" + "=" * 70)

        except KeyboardInterrupt:
            print("\nExiting.")
            break

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()
