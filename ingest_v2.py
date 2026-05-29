import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
import chromadb


def build_semantic_database(pdf_path: str):
    print("1. Extracting PDF to Markdown (Preserving Tables)...")
    # This magically converts PDF tables into format like | Co-Pay | 10% |
    md_text = pymupdf4llm.to_markdown(pdf_path)

    print("2. Chunking semantically by Headers...")
    # Tell the splitter to break chunks ONLY when it sees a new section header
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(md_text)

    print(f"Created {len(md_header_splits)} highly-contextual chunks.")

    print("3. Pushing to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Reset the old database collection to prevent overlap
    try:
        chroma_client.delete_collection(name="policy_docs")
    except:
        pass

    collection = chroma_client.create_collection(name="policy_docs")

    documents = []
    metadatas = []
    ids = []

    for i, split in enumerate(md_header_splits):
        documents.append(split.page_content)

        # --- THE FIX ---
        # Grab the existing metadata (if any)
        meta = split.metadata
        # Force a 'source' key so the dictionary is NEVER empty
        meta["source"] = "hdfc_optima_restore"

        metadatas.append(meta)
        ids.append(f"chunk_{i}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("✅ Semantic Database successfully built!")


# Run it on your specific HDFC document
if __name__ == "__main__":
    build_semantic_database("app/data/policies/HDFHLIP25012V082425.PDF")