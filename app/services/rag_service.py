import os
import chromadb
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. Ensure the unified policy directory exists
POLICY_DIR = "app/data/policies"
os.makedirs(POLICY_DIR, exist_ok=True)

# 2. Establish ONE global database connection
# This matches exactly where ai_service.py is looking!
chroma_client = chromadb.PersistentClient(path="app/data/chroma_db")
collection = chroma_client.get_or_create_collection(name="policy_docs")


def append_pdf_to_database(pdf_path: str, filename: str):
    """Reads a PDF, converts to Markdown, and semantically chunks it."""
    print(f"Extracting and chunking {filename}...")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
        )
        md_header_splits = markdown_splitter.split_text(md_text)

        documents, metadatas, ids = [], [], []
        for i, split in enumerate(md_header_splits):
            documents.append(split.page_content)

            # Ensure metadata always has a source to prevent ChromaDB crash
            meta = split.metadata if split.metadata else {}
            meta["source"] = filename
            metadatas.append(meta)

            # Use filename and chunk index for stable, unique IDs
            ids.append(f"{filename}_chunk_{i}")

        # --- THE FIX: Use UPSERT instead of ADD ---
        # Upsert will add new chunks, or cleanly overwrite existing ones if the server reboots.
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ Successfully synced {filename} to ChromaDB ({len(documents)} chunks).")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")


def ingest_policies():
    """Runs on server boot: Syncs all PDFs in the policy folder to the Vector DB."""
    files = [f for f in os.listdir(POLICY_DIR) if f.lower().endswith(".pdf")]

    if not files:
        print("No policy files found in the 'policy/' directory to ingest.")
        return

    print(f"Found {len(files)} files. Syncing to database...")
    for filename in files:
        filepath = os.path.join(POLICY_DIR, filename)
        # Re-use our beautiful markdown chunker for server bootups!
        append_pdf_to_database(filepath, filename)

# Hello Docker!