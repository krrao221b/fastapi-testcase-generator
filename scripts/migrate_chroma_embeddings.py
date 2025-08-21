"""Migration helper: copy embeddings from Chroma metadata into SQLite BLOB column.

Run with: python scripts/migrate_chroma_embeddings.py

This script requires the app settings to be configured and will:
- open a DB session
- iterate Chroma collection metadatas
- for each metadata with "test_case_id" and "embedding", parse JSON embedding and write to TestCaseModel.embedding_vector as float32 bytes
"""
import json
import numpy as np
from app.config.settings import settings
import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base, TestCaseModel

# Configure DB
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Initialize chroma
client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
collection = client.get_collection(settings.chroma_collection_name)


def migrate():
    db = SessionLocal()
    try:
        for meta in collection.get(include=['metadatas'])['metadatas']:
            for m in meta:
                try:
                    tc_id = int(m.get('test_case_id'))
                    emb_json = m.get('embedding')
                    if emb_json:
                        emb = json.loads(emb_json)
                        arr = np.array(emb, dtype=np.float32)
                        db_case = db.query(TestCaseModel).filter(TestCaseModel.id == tc_id).first()
                        if db_case:
                            db_case.embedding_vector = arr.tobytes()
                            print(f"Migrated embedding for test_case {tc_id}")
                except Exception as e:
                    print(f"Skipping metadata entry due to error: {e}")
        db.commit()
    finally:
        db.close()


if __name__ == '__main__':
    migrate()
    print("Migration complete.")
