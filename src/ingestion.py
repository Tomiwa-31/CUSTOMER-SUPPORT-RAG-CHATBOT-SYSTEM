# src/ingestion.py

import os
import io
import pickle
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    DATA_DIR,
    NAMESPACE_DOC_MAP,
    DB_NAMESPACE,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HEADERS_TO_SPLIT_ON,
    BM25_TOP_K,
    PINECONE_INDEX_NAME,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────
EMBEDDING_DIMENSION = 384   # all-MiniLM-L6-v2 output dimension
GCS_BM25_PREFIX     = "bm25"   # gs://your-bucket/bm25/bm25_{namespace}.pkl
 

# ── BM25 path helper ──────────────────────────────────────────────────────
def get_bm25_path(namespace: str) -> str:  ###file storage path
    return f"./bm25_{namespace}.pkl"

def get_bm25_gcs_blob(namespace: str) -> str:    ###cloud storage path
    """GCS object path for a BM25 pickle, e.g. bm25/bm25_billing_support.pkl"""
    return f"{GCS_BM25_PREFIX}/bm25_{namespace}.pkl"

# ── Embedding model ───────────────────────────────────────────────────────
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )


# ── Pinecone index setup ──────────────────────────────────────────────────
def get_or_create_pinecone_index():
    """Create the Pinecone index if it doesn't already exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Created Pinecone index: '{PINECONE_INDEX_NAME}'")
    else:
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists")

    return pc


# ── Namespace existence check & Namespace helper ─────────────────────────────────────────────
def namespace_has_vectors(pc: Pinecone, namespace: str) -> bool:
    """Check if a namespace already has vectors — skip if so."""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        count = namespaces.get(namespace, {}).get("vector_count", 0)
        return count > 0
    except Exception:
        return False

def delete_namespace(pc: Pinecone, namespace: str):
    """
    Wipe all vectors in a namespace so we can do a clean re-ingest.
    Used for northwind_db on every scheduled run.
    """
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
        print(f"  🗑️  Wiped Pinecone namespace '{namespace}'")
    except Exception as e:
        print(f"  ⚠️  Could not wipe namespace '{namespace}': {e}")


# ── GCS helpers ────────────────────────────────────────────────────────────
def upload_bm25_to_gcs(namespace: str):
    """
    Upload the local BM25 pickle for a namespace to GCS.
    Destination: gs://{GCS_BUCKET_NAME}/bm25/bm25_{namespace}.pkl
    """
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is not set")
 
    local_path = get_bm25_path(namespace)
    blob_name  = get_bm25_gcs_blob(namespace)# a file in a bucket is called a blob
 
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
 
    blob.upload_from_filename(local_path)
    print(f"  ☁️  BM25 uploaded → gs://{bucket_name}/{blob_name}")
 
 
def download_bm25_from_gcs(namespace: str) -> BM25Retriever:
    """
    Download the BM25 pickle for a namespace from GCS and return
    the loaded BM25Retriever. Called by the app at startup instead
    of reading a local file.
    """
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is not set")
 
    blob_name = get_bm25_gcs_blob(namespace)
 
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
 
    data = blob.download_as_bytes()
    retriever = pickle.loads(data)
    retriever.k = BM25_TOP_K
 
    print(f"  ☁️  BM25 loaded ← gs://{bucket_name}/{blob_name}")
    return retriever
 


# ── File-based loader ─────────────────────────────────────────────────────
def load_documents_for_collection(collection_name: str) -> list:
    """Load docs assigned to this collection from flat files."""
    docs = []
    filenames = NAMESPACE_DOC_MAP.get(collection_name, [])

    for filename in filenames:
        filepath = Path(DATA_DIR) / filename
        if not filepath.exists():
            print(f"  ⚠️  File not found: {filepath}")
            continue
        loader = TextLoader(str(filepath))
        docs.extend(loader.load())
        #source field is literally just the file path you passed in

    print(f"  📄 Loaded {len(docs)} file documents for '{collection_name}'")
    return docs


# ── Database loader ───────────────────────────────────────────────────────
def load_documents_from_database() -> list:
    """
    Query Northwind tables from Neon PostgreSQL and return
    a list of rich, joined LangChain Documents.
    """
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        documents = []
    except Exception as e:
        print(f"  ❌ Error connecting to database: {e}")
        return []

    # ── Products ──────────────────────────────────────────────────────────
    print("  📦 Loading products...")
    cursor.execute("""
        SELECT
            p.product_id,
            p.product_name,
            p.unit_price,
            p.units_in_stock,
            p.units_on_order,
            p.discontinued,
            p.quantity_per_unit,
            c.category_name,
            c.description     AS category_description,
            s.company_name    AS supplier_name,
            s.country         AS supplier_country
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
        JOIN suppliers  s ON p.supplier_id  = s.supplier_id
    """)

    for row in cursor.fetchall():
        (product_id, name, price, in_stock, on_order,
         discontinued, qty_per_unit, category, cat_desc,
         supplier, supplier_country) = row

        status = "discontinued" if discontinued else "available"

        text = (
            f"Product: {name}\n"
            f"Category: {category} — {cat_desc}\n"
            f"Price: ${price} per unit ({qty_per_unit})\n"
            f"Stock: {in_stock} units in stock, {on_order} units on order\n"
            f"Status: {status}\n"
            f"Supplier: {supplier} from {supplier_country}"
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source":       "northwind_db",
                "type":         "product",
                "product_id":   str(product_id),
                "product_name": name,
                "category":     category,
            }
        ))

    # ── Orders ────────────────────────────────────────────────────────────
    print("  🛒 Loading orders...")
    cursor.execute("""
        SELECT
            o.order_id,
            o.order_date,
            o.shipped_date,
            o.freight,
            o.ship_country,
            o.ship_city,
            c.company_name                     AS customer_name,
            c.country                          AS customer_country,
            e.first_name || ' ' || e.last_name AS employee_name,
            s.company_name                     AS shipper_name
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN employees e ON o.employee_id  = e.employee_id
        JOIN shippers  s ON o.ship_via     = s.shipper_id
    """)

    orders = cursor.fetchall()

    for order in orders:
        (order_id, order_date, shipped_date, freight,
         ship_country, ship_city, customer_name,
         customer_country, employee_name, shipper_name) = order

        cursor.execute("""
            SELECT
                p.product_name,
                od.quantity,
                od.unit_price,
                od.discount
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            WHERE od.order_id = %s
        """, (order_id,))

        items = cursor.fetchall()
        items_text = "\n".join([
            f"  - {qty}x {name} at ${price:.2f} "
            f"(discount: {int(disc * 100)}%)"
            for name, qty, price, disc in items
        ])
        total = sum(
            qty * price * (1 - disc) for _, qty, price, disc in items
        )

        text = (
            f"Order #{order_id} placed on {order_date}\n"
            f"Customer: {customer_name} from {ship_city}, "
            f"{customer_country}\n"
            f"Handled by: {employee_name}\n"
            f"Shipped via: {shipper_name} to {ship_city}, "
            f"{ship_country}\n"
            f"Shipped date: {shipped_date or 'not yet shipped'}\n"
            f"Freight cost: ${freight}\n"
            f"Order total: ${total:.2f}\n"
            f"Items ordered:\n{items_text}"
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source":        "northwind_db",
                "type":          "order",
                "order_id":      str(order_id),
                "customer_name": customer_name,
                "ship_country":  ship_country,
                "order_date":    str(order_date),
            }
        ))

    # ── Customers ─────────────────────────────────────────────────────────
    print("  👤 Loading customers...")
    cursor.execute("""
        SELECT
            c.customer_id,
            c.company_name,
            c.contact_name,
            c.contact_title,
            c.city,
            c.country,
            c.phone,
            COUNT(o.order_id) AS total_orders,
            COALESCE(
                SUM(od.unit_price * od.quantity * (1 - od.discount)), 0
            )                 AS total_spent
        FROM customers c
        LEFT JOIN orders       o  ON c.customer_id = o.customer_id
        LEFT JOIN order_details od ON o.order_id   = od.order_id
        GROUP BY
            c.customer_id, c.company_name, c.contact_name,
            c.contact_title, c.city, c.country, c.phone
    """)

    for row in cursor.fetchall():
        (customer_id, company, contact, title,
         city, country, phone, total_orders, total_spent) = row

        text = (
            f"Customer: {company}\n"
            f"Contact: {contact} ({title})\n"
            f"Location: {city}, {country}\n"
            f"Phone: {phone}\n"
            f"Total orders placed: {total_orders}\n"
            f"Total lifetime spend: ${float(total_spent):.2f}"
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source":       "northwind_db",
                "type":         "customer",
                "customer_id":  customer_id,
                "company_name": company,
                "country":      country,
            }
        ))

    cursor.close()
    conn.close()

    total = len(documents)
    n_products  = sum(1 for d in documents if d.metadata["type"] == "product")
    n_orders    = sum(1 for d in documents if d.metadata["type"] == "order")
    n_customers = sum(1 for d in documents if d.metadata["type"] == "customer")

    print(f"✅ Database documents loaded: {total} total")
    print(f"   Products:  {n_products}")
    print(f"   Orders:    {n_orders}")
    print(f"   Customers: {n_customers}")
    return documents


# ── Chunking ──────────────────────────────────────────────────────────────
#SEPERATE CHUNKING STRATEGY BY SOURCE
def chunk_file_documents(docs: list) -> list:
    """
    File docs go through markdown splitting first,
    then recursive splitting — same as your original logic.
    """
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_splits = []
    for doc in docs:
        md_splits = markdown_splitter.split_text(doc.page_content)
        further_splits = text_splitter.split_documents(md_splits)
        all_splits.extend(further_splits)

    print(f"  ✅ Created {len(all_splits)} chunks")
    return all_splits


def chunk_db_documents(docs: list) -> list:
    """
    DB docs are already clean structured text — skip markdown
    splitting and go straight to recursive splitting.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(docs)
    print(f"  ✅ Created {len(splits)} chunks")
    return splits


# ── Pinecone upsert ───────────────────────────────────────────────────────
def build_vectorstore(
    splits: list,
    namespace: str,
    embedding_model: HuggingFaceEmbeddings
):
    """Upsert document chunks into Pinecone under the given namespace."""
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embedding_model,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )
    print(f"  ✅ Upserted {len(splits)} chunks → "
          f"Pinecone namespace '{namespace}'")


# ── BM25 ──────────────────────────────────────────────────────────────────
def build_bm25_index(splits: list, namespace: str):
    """Build and persist a BM25 index for a given namespace."""
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = BM25_TOP_K

    bm25_path = get_bm25_path(namespace)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"  ✅ BM25 index saved → {bm25_path}")
    upload_bm25_to_gcs(namespace)
    return bm25_retriever


# ── Main runner ───────────────────────────────────────────────────────────
def run_ingestion():
    embedding_model = load_embedding_model()
    pc = get_or_create_pinecone_index()

    # ── File-based collections ─────────────────────────────────────────────
    for namespace in NAMESPACE_DOC_MAP:
        print(f"\n📂 File collection: '{namespace}'")

        if namespace_has_vectors(pc, namespace=namespace):
            print(f"  ⏭️  Already ingested — skipping")
            continue

        docs = load_documents_for_collection(namespace)
        if not docs:
            print(f"  ⚠️  No documents found — skipping")
            continue

        splits = chunk_file_documents(docs)
        build_vectorstore(splits, namespace=namespace,
                          embedding_model=embedding_model)
        build_bm25_index(splits, namespace=namespace)

    # ── Northwind database ─────────────────────────────────────────────────
    print(f"\n🗄️  Database collection: 'northwind_db'")

    if namespace_has_vectors(pc, namespace="northwind_db"):
        print(f"  ⏭️  Already ingested — skipping")
    else:
        db_docs = load_documents_from_database()
        if db_docs:
            splits = chunk_db_documents(db_docs)
            build_vectorstore(splits, namespace="northwind_db",
                              embedding_model=embedding_model)
            build_bm25_index(splits, namespace="northwind_db")

    print("\n✅ All ingestion complete!")


if __name__ == "__main__":
    run_ingestion()

#so i curretly have 2 vector stores in pinecone, one for file-based docs and one for db docs, where the file based has 3 namespace and the db has 1 namespace, each with its own bm25 index saved as a pickle file locally. The retriever will then load the relevant namespaces based on the question asked and use both the bm25 and vector search to retrieve the top k relevant chunks to feed into the llm.