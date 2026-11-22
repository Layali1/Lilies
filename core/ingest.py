"""PDF -> clean text -> chunks -> Chroma. Keeps page numbers for citations."""

"""Small PDF ingestion pipeline:
PDF -> extracted text -> cleaned pages -> chunks -> Chroma.
Keeps page numbers for citations.
"""

# ============================================================
# 1. IMPORTS — استيراد الأدوات التي نحتاجها
# ============================================================

from pypdf import PdfReader
# قراءة ملفات PDF واستخراج النص من صفحاتها

from langchain_core.documents import Document
# إنشاء مستند يحتوي على النص (page_content) والبيانات الإضافية (metadata)

from langchain_text_splitters import RecursiveCharacterTextSplitter
# تقسيم النصوص الكبيرة إلى أجزاء صغيرة (Chunks)

from langchain_chroma import Chroma
# التعامل مع قاعدة البيانات المتجهية Chroma

from langchain_huggingface import HuggingFaceEmbeddings
# استخدام نماذج Hugging Face لتحويل النصوص إلى Embeddings

from core.utils import clean
# استيراد دالة التنظيف من ملف utils.py داخل مجلد core


# ============================================================
# 2. CONFIGURATION — إعدادات النموذج والتقسيم
# ============================================================

EMBED_MODEL = "intfloat/multilingual-e5-small"
# نموذج Embeddings متعدد اللغات يدعم العربية والإنجليزية

EMBED_DIM = 384
# عدد الأبعاد في المتجه الناتج من النموذج

CHUNK_SIZE, CHUNK_OVERLAP = 800, 120
# CHUNK_SIZE: الحجم المستهدف لكل Chunk (بالحروف)
# CHUNK_OVERLAP: مقدار التداخل بين الأجزاء المتجاورة


# ============================================================
# 3. E5 EMBEDDINGS — تخصيص طريقة تحويل النصوص إلى متجهات
# ============================================================

class E5Embeddings(HuggingFaceEmbeddings):
    """e5 models expect 'passage:' for documents and 'query:' for questions."""

    def embed_documents(self, texts):
        # نضيف passage: في بداية كل نص نريد تخزينه وفهرسته
        # ثم نستخدم دالة الأب لتحويل النصوص إلى Embeddings

        return super().embed_documents(
            [f"passage: {t}" for t in texts]
        )

    def embed_query(self, text):
        # نضيف query: في بداية سؤال المستخدم
        # نستخدم دالة الأب مباشرة لتجنب إضافة passage: مرة ثانية
        # [0] لاستخراج المتجه الوحيد من القائمة الناتجة

        return super().embed_documents([f"query: {text}"])[0]


# ============================================================
# 4. MAKE EMBEDDINGS — إنشاء كائن النموذج
# ============================================================

def make_embeddings() -> E5Embeddings:

    return E5Embeddings(
        model_name=EMBED_MODEL,
        # تحديد نموذج Embeddings المستخدم

        encode_kwargs={"normalize_embeddings": True}
        # تطبيع المتجهات الناتجة
    )


# ============================================================
# 5. GET STORE — إنشاء مخزن Chroma
# ============================================================

def get_store(path: str = "chroma_db", embeddings=None) -> Chroma:

    return Chroma(
        collection_name="lilies",
        # اسم مجموعة البيانات داخل Chroma

        embedding_function=embeddings or make_embeddings(),
        # استخدام Embeddings الممرر للدالة
        # أو إنشاء واحد جديد إذا لم يتم تمريره

        persist_directory=path
        # مكان تخزين بيانات Chroma محليًا
    )


# ============================================================
# 6. SPLIT PAGES — تقسيم الصفحات إلى Chunks
# ============================================================

def split_pages(pages: list) -> list:

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,
        # الحجم المستهدف لكل جزء

        chunk_overlap=CHUNK_OVERLAP,
        # مقدار التداخل بين الأجزاء

        separators=["\n\n", "\n", ".", "!", "?", "؟", "،", "؛", ";", " ", ""]
        # ترتيب الفواصل التي يحاول المقسّم استخدامها:
        # فقرة، سطر جديد، نقطة، استفهام، فاصلة، مسافة
    )

    # تقسيم المستندات وإرجاع قائمة المقاطع الناتجة
    return splitter.split_documents(pages)


# ============================================================
# 7. ADD CHUNKS — إضافة المقاطع إلى Chroma
# ============================================================

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


# ============================================================
# 8. ADD PDF — الدالة الرئيسية لمعالجة ملف PDF
# ============================================================

def add_pdf(store: Chroma, file, name: str, progress=None) -> int:
    reader = PdfReader(file)
    labels = reader.page_labels  # printed page numbers, e.g. "317"
    pages = [Document(page_content=clean(p.extract_text() or ""),
                      metadata={"source": name, "page": i + 1, "label": labels[i]})
             for i, p in enumerate(reader.pages)]
    pages = [p for p in pages if p.page_content]
    return add_chunks(store, split_pages(pages), progress)

