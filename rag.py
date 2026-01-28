# ==========================================================
# STABLE RAG FOR SDN ATTACK ANALYSIS (RYU / ODL)
# Threat-Intelligence-Oriented (NO mitigation)
# ==========================================================

import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

BASE_LOG_DIR = "/home/akhil/Desktop/Attacks/Ryu"   # change if needed
VECTOR_DB_DIR = "./sdn_vector_db"
COLLECTION_NAME = "sdn_attack_knowledge"

# ✅ Supported Groq model (70B removed by Groq)
MODEL_NAME = "llama-3.1-8b-instant"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------
# SIMPLE TEXT SPLITTER (LOG SAFE)
# ----------------------------------------------------------

def simple_split(text, max_len=500):
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
# LOAD SINGLE CONTROLLER LOG
# ----------------------------------------------------------

def load_controller_log(path, attack, controller="ryu"):
    docs = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            text = f"""
ATTACK: {attack}
CONTROLLER: {controller}
PLANE: CONTROL
LOG: {line}
"""

            for chunk in simple_split(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "attack": attack,
                            "controller": controller,
                            "plane": "control"
                        }
                    )
                )

    return docs

# ----------------------------------------------------------
# LOAD ALL LOGS FROM DIRECTORY
# ----------------------------------------------------------

def load_all_logs(base_dir):
    all_docs = []

    for file in os.listdir(base_dir):
        if not file.endswith(".log"):
            continue

        full = os.path.join(base_dir, file)
        name = file.lower()

        attack = "unknown"
        if "dos" in name:
            attack = "dos"
        elif "arp" in name:
            attack = "arp_spoofing"
        elif "flow" in name:
            attack = "flow_rule_poisoning"

        print(f"[+] LOG: {full}")
        all_docs += load_controller_log(full, attack)

    return all_docs

# ----------------------------------------------------------
# RAG QUESTION ANSWERING (NO CHAINS)
# ----------------------------------------------------------

def ask_rag(llm, retriever, query):
    # ✅ Correct retriever call (new API)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant evidence found in controller logs."

    context = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )

    prompt = f"""
You are an SDN security researcher.

Answer ONLY using the provided controller logs.
Do NOT guess.
If evidence is missing, explicitly say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return llm.invoke(prompt).content

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    print("[+] Loading controller logs")
    documents = load_all_logs(BASE_LOG_DIR)

    if not documents:
        print("[!] No logs found.")
        return

    print(f"[+] Total log chunks indexed: {len(documents)}")

    embeddings = FastEmbedEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
            print("\nANSWER:\n", ans)
            print("\n" + "=" * 70)

        except KeyboardInterrupt:
            print("\nExiting.")
            break

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()
