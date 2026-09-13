# MediAssist AI - app.py
"""
Step 1: Retrieve PMIDs from PubMed (run once via init)
Step 2: Create a ChromaDB vector collection
Step 3: Fetch full PubMed articles from PMIDs.
Step 4 — Create embeddings for articles and store them in ChromaDB.
Step 5: Query ChromaDB collection with biomedical embeddings. Uses the same PubMedBERT-based model for semantic similarity.
Step 6 — RAG pipeline using ChromaDB for retrieval + Groq (openai/gpt-oss-20b) for generation.
"""
import json
import os
try:
    from pubmed import PubMedRetriever
except ImportError:
    from src.pubmed import PubMedRetriever
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
dotenv_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

collection_name = "pubmed_if_articles"

# Initialize Chroma client ONCE for the entire lifecycle
chroma_path = os.path.join(DATA_DIR, "chroma_data") if os.path.exists(os.path.join(DATA_DIR, "chroma_data")) else "./chroma_data"
client = chromadb.PersistentClient(path=chroma_path)

# 🧠 Initialize embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# 🔹 Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def fetch_pmids():
    """
    🩺 PubMed Retriever
    Step 1 — Retrieve up to 300 PMIDs from PubMed.
    """
    # 🧬 Search PubMed for intermittent fasting + related conditions
    pmids = PubMedRetriever.search_pubmed_articles(
        # search_term="intermittent fasting AND (obesity OR \"type 2 diabetes\" OR metabolic disorders)",
        search_term="fasting glucose AND intermittent fasting",
        max_results=300
    )

    # 📚 Print result count
    print(f"Retrieved {len(pmids)} PMIDs from PubMed.")

    # Optional: save PMIDs to file
    pmids_path = os.path.join(DATA_DIR, "pmids.txt") if os.path.exists(DATA_DIR) else "pmids.txt"
    with open(pmids_path, "w", encoding="utf-8") as f:
        for pmid in pmids:
            f.write(f"{pmid}\n")
    print(f"PMIDs saved to {pmids_path}")


def create_or_get_pmids_collection_in_chromadb():
    """
    🧠 Import ChromaDB
    Step 2 — Create a ChromaDB collection to store PubMed article vectors.
    """
    # 📦 Create the collection (only if it does not exist)
    try:
        collection = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")

    return collection


def fetch_full_articles():
    """
    Step 3 — Fetch full article metadata for PMIDs from PubMed.
    Reads PMIDs from pmids.txt (Step 1 output).
    Saves detailed article data to articles.json.
    """
    pmid_file = os.path.join(DATA_DIR, "pmids.txt") if os.path.exists(os.path.join(DATA_DIR, "pmids.txt")) else "pmids.txt"
    output_file = os.path.join(DATA_DIR, "articles.json") if os.path.exists(DATA_DIR) else "articles.json"

    # 📚 Read PMIDs from file
    try:
        with open(pmid_file, "r", encoding="utf-8") as f:
            pmids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {pmid_file} not found. Please run Step 1 first.")
        return

    if not pmids:
        print("No PMIDs found in file.")
        return

    print(f"Fetching {len(pmids)} articles from PubMed...")

    # 🔎 Fetch full articles
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)

    # 💾 Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(articles)} articles.")
    print(f"Saved details to {output_file}")


def store_articles_embeddings_in_chromadb():
    """
    Step 4 — Create embeddings for articles and store them in ChromaDB.
    Reads articles.json (from Step 3) and ingests into existing collection.
    """
    articles_file = os.path.join(DATA_DIR, "articles.json") if os.path.exists(os.path.join(DATA_DIR, "articles.json")) else "articles.json"

    # 📚 Load articles
    try:
        with open(articles_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {articles_file} not found. Please run Step 3 first.")
        return

    if not articles:
        print("⚠ No articles found in file.")
        return

    # 📦 Get or create the collection
    try:
        collection = create_or_get_pmids_collection_in_chromadb()
    except Exception:
        collection = client.create_collection(name=collection_name)

    print(f"📥 Ingesting {len(articles)} articles into '{collection_name}'...")

    # 🔬 Prepare data
    ids = []
    docs = []
    metas = []

    for article in articles:
        pmid = article.get("pmid", "")
        title = article.get("title", "No Title")
        abstract_text = " ".join(article.get("abstract", {}).values()) if article.get("abstract") else "No Abstract"

        ids.append(pmid)
        docs.append(f"{title}. {abstract_text}")
        metas.append({
            "pmid": pmid,
            "title": title,
            "journal": article.get("journal", "Unknown Journal"),
            "authors": ", ".join(article.get("authors", [])) if isinstance(article.get("authors"), list) else article.get("authors", "No Authors"),
            "publication_date": article.get("publication_date", "Unknown Date")
        })

    # 🧬 Generate embeddings
    embeddings = model.encode(docs, show_progress_bar=True)

    # 💾 Add to ChromaDB
    collection.add(
        ids=ids,
        documents=docs,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metas
    )

    print(f"✅ Ingested {len(ids)} articles into '{collection_name}'.")


def query_chromadb():
    """
    Step 5 — Query the ChromaDB collection using semantic search.
    Prompts the user for a medical question and returns top matching articles.
    """

    # 📦 Get the collection
    try:
        collection = create_or_get_pmids_collection_in_chromadb()
    except Exception:
        print(f"Error: Collection '{collection_name}' not found. Please run Step 4 first.")
        return

    print("\n=== MediAssist AI — Semantic Search ===")
    print("Type your medical question below (or 'exit' to quit):")

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting search.")
            break

        if not query:
            print("Please enter a question.")
            continue

        # 🔍 Embed the query
        query_embedding = model.encode([query])

        # 📚 Search top 5 results
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5,
            include=["metadatas", "documents"]
        )

        # 📝 Display results
        print("\nTop matching articles:")
        for i, (pmid, meta, doc) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["documents"][0]
        ), start=1):
            print(f"\n{i}. PMID: {pmid}")
            print(f"   Title: {meta.get('title', 'No Title')}")
            print(f"   Journal: {meta.get('journal', 'Unknown Journal')}")
            print(f"   Authors: {meta.get('authors', 'No Authors')}")
            print(f"   Date: {meta.get('publication_date', 'Unknown Date')}")
            snippet = doc[:300].replace("\n", " ") + "..."
            print(f"   Snippet: {snippet}")


def answer_query_with_rag_groq(question):
    """
    Step 6 — RAG pipeline using ChromaDB for retrieval + Groq (openai/gpt-oss-20b) for generation.
    """
    # 1️⃣ Retrieve top documents from ChromaDB
    # 📦 Get or create the collection
    try:
        collection = create_or_get_pmids_collection_in_chromadb()
    except Exception:
        collection = client.create_collection(name=collection_name)

    query_embedding = model.encode([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "metadatas"]
    )

    # 📝 Display results
    print("\nTop matching articles:")
    for i, (pmid, meta, doc) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["documents"][0]
    ), start=1):
        print(f"\n{i}. PMID: {pmid}")
        print(f"   Title: {meta.get('title', 'No Title')}")
        print(f"   Journal: {meta.get('journal', 'Unknown Journal')}")
        print(f"   Authors: {meta.get('authors', 'No Authors')}")
        print(f"   Date: {meta.get('publication_date', 'Unknown Date')}")
        snippet = doc[:300].replace("\n", " ") + "..."
        print(f"   Snippet: {snippet}")

    # 2️⃣ Build context from retrieved docs
    context = ""
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        context += f"- Title: {meta.get('title', 'No Title')} ({meta.get('publication_date', 'Unknown Date')})\n"
        context += f"  Journal: {meta.get('journal', 'Unknown Journal')}\n"
        context += f"  Authors: {meta.get('authors', 'No Authors')}\n"
        context += f"  Snippet: {doc[:400]}...\n\n"

    # 3️⃣ Create RAG prompt
    prompt = f"""
You are a medical research assistant for doctors.
Answer the question below using ONLY the provided research context.
If the context does not contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:
"""

    # 📝 Display results
    print("\n\n\n\nprompt")

    # 4️⃣ Generate response from Groq (openai/gpt-oss-20b)
    chat_completion = groq_client.chat.completions.create(
        model=os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b'),
        messages=[
            {"role": "system", "content": "You are a helpful and evidence-based medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    # 5️⃣ Output answer
    final_answer = chat_completion.choices[0].message.content.strip()
    print("\n=== Final Answer (Groq openai/gpt-oss-20b) ===\n")
    print(final_answer)
    return final_answer


# Run once
if __name__ == "__main__":
    # fetch_pmids()
    # create_or_get_pmids_collection_in_chromadb()
    # fetch_full_articles()
    # client.delete_collection("pubmed_if_articles")
    # store_articles_embeddings_in_chromadb()
    # query_chromadb()
    # answer_query_with_rag_groq("What are the benefits of intermittent fasting for type 2 diabetes?")
    answer_query_with_rag_groq("Is alternate day fasting effective for obesity?")
