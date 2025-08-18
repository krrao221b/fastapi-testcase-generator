"""
Reindex embeddings for all test cases using the currently configured embedding model.
This will:
 - delete existing chroma documents for each test case
 - generate new embeddings with current model
 - store updated embedding blob in SQLite and add documents to Chroma

Usage:
    python scripts/reindex_embeddings.py

Make sure your environment variables for OpenAI are set and the server config is correct.
"""
import asyncio
from app.core.database import SessionLocal
from app.models.database import TestCaseModel
from app.repositories.implementations.chroma_memory_service import ChromaMemoryService


def main():
    db = SessionLocal()
    try:
        svc = ChromaMemoryService(db)
        cases = db.query(TestCaseModel).all()
        print(f"Found {len(cases)} test cases to reindex")

        async def run():
            for c in cases:
                # Build a minimal TestCase pydantic-like object expected by service
                from app.models.schemas import TestCase as TC
                tc = TC.model_validate(c)
                print(f"Reindexing test case id={c.id} title={c.title}")
                ok = await svc.update_test_case_embedding(tc)
                print(f" -> success={ok}")

        asyncio.run(run())
    finally:
        db.close()


if __name__ == '__main__':
    main()
