from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from chromadb.config import Settings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

docs = [
    Document(page_content="This is a test doc about admissions", metadata={"topic": "admissions"}),
    Document(page_content="This is a test doc about CAS program", metadata={"topic": "CAS"})
]

print("Building test Chroma...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="test_collection",
    client_settings=Settings(
        anonymized_telemetry=False,
        persist_directory="./chroma_test_db",
        is_persistent=True
    )
)
print("Chroma built successfully, persisting...")
vectorstore.persist()
print("Persisted! Total docs:", vectorstore._collection.count())
