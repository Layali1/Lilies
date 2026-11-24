"""PDF -> clean text -> chunks -> Chroma. Keeps page numbers for citations."""

"""Small PDF ingestion pipeline:
PDF -> extracted text -> cleaned pages -> chunks -> Chroma.
Keeps page numbers for citations.
"""


from pypdf import PdfReader

from langchain_core.documents import Document


from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_chroma import Chroma


from langchain_huggingface import HuggingFaceEmbeddings


from core.utils import clean




EMBED_MODEL = "intfloat/multilingual-e5-small"

EMBED_DIM = 384

CHUNK_SIZE, CHUNK_OVERLAP = 800, 120

class E5Embeddings(HuggingFaceEmbeddings):
    """e5 models expect 'passage:' for documents and 'query:' for questions."""

    def embed_documents(self, texts):
       
        return super().embed_documents(
            [f"passage: {t}" for t in texts]
        )

    def embed_query(self, text):
        
        return super().embed_documents([f"query: {text}"])[0]



def make_embeddings() -> E5Embeddings:

    return E5Embeddings(
        model_name=EMBED_MODEL,
        
        encode_kwargs={"normalize_embeddings": True}
        
    )



def get_store(path: str = "chroma_db", embeddings=None) -> Chroma:

    return Chroma(
        collection_name="lilies",
        

        embedding_function=embeddings or make_embeddings(),
        
        persist_directory=path

    )




def split_pages(pages: list) -> list:

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,
        
        chunk_overlap=CHUNK_OVERLAP,
       
        separators=["\n\n", "\n", ".", "!", "?", "؟", "،", "؛", ";", " ", ""]
       
    )

    
    return splitter.split_documents(pages)




def add_chunks(store: Chroma, chunks: list, progress=None, batch_size: int = 64) -> int:
    """Adds in batches, skips chunks already stored, reports progress (0..1)."""
    counter, ids = {}, []
    for c in chunks:
        key = f"{c.metadata['source']}:{c.metadata['page']}"
        counter[key] = counter.get(key, 0) + 1
        ids.append(f"{key}:{counter[key]}")
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = list(zip(chunks[start:start + batch_size], ids[start:start + batch_size]))
        existing = set(store.get(ids=[i for _, i in batch])["ids"])
        todo = [(c, i) for c, i in batch if i not in existing]
        if todo:
            store.add_documents([c for c, _ in todo], ids=[i for _, i in todo])
        if progress:
            progress(min(start + batch_size, total) / total)
    return total




def add_pdf(store: Chroma, file, name: str, progress=None) -> int:
    reader = PdfReader(file)
    labels = reader.page_labels  # printed page numbers, e.g. "317"
    pages = [Document(page_content=clean(p.extract_text() or ""),
                      metadata={"source": name, "page": i + 1, "label": labels[i]})
             for i, p in enumerate(reader.pages)]
    pages = [p for p in pages if p.page_content]
    return add_chunks(store, split_pages(pages), progress)

