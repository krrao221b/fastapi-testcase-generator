import chromadb
from typing import List
from app.repositories.interfaces.memory_service import IMemoryService
from app.models.schemas import TestCase, SimilarTestCase
from app.config.settings import settings
from app.models.database import TestCaseModel
from sqlalchemy.orm import Session
import structlog
import numpy as np
import json
from openai import OpenAI
import traceback

logger = structlog.get_logger()

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
)

# Define embedding model
embedding_model = settings.openai_embedding_model


class ChromaMemoryService(IMemoryService):
    """ChromaDB implementation of memory service with embedding and SQLite sync"""

    def __init__(self, db_session: Session):
        logger.info(
            "Initializing ChromaMemoryService",
            chroma_path=settings.chroma_persist_directory,
            collection=settings.chroma_collection_name,
        )
        try:
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            # No embedding function bound to collection; we pass explicit vectors and control provider.
            self.embedding_function = None
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            self.db = db_session
            # Attempt to infer embedding dimension once for defensive checks
            self.embedding_dim = self._determine_embedding_dim()
            logger.info(
                "ChromaMemoryService initialized", embedding_dim=self.embedding_dim
            )
        except Exception as e:
            logger.error("Failed to initialize ChromaMemoryService", error=str(e))
            logger.debug(traceback.format_exc())
            raise

    def _determine_embedding_dim(self) -> int | None:
        # Offline map only to avoid startup network calls that can hang
        model_dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        dim = model_dim_map.get(embedding_model, None)
        if dim is None:
            logger.warning(
                "Unknown embedding model; embedding_dim left as None",
                model=embedding_model,
            )
        return dim

    def _get_embedding(self, text: str) -> List[float]:
        # Fast guards
        if not settings.openai_api_key:
            logger.warning("No OPENAI API key configured; skipping embeddings")
            return []
        try:
            client = openai_client.with_options(timeout=8.0)
            resp = client.embeddings.create(model=embedding_model, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            logger.error("Embedding API error", error=str(e))
            logger.debug(traceback.format_exc())
            return []

    async def store_test_case(self, test_case: TestCase) -> bool:
        try:
            combined_text = (
                f"Title: {test_case.title}\n"
                f"Description: {test_case.description}\n"
                f"Feature: {test_case.feature_description}\n"
                f"Acceptance: {test_case.acceptance_criteria}\n"
                f"Tags: {', '.join(test_case.tags)}\n"
                f"Priority: {test_case.priority.value}"
            )
            embedding = self._get_embedding(combined_text)
            if not embedding:
                return True  # do not fail the flow on embedding issues

            # Full doc
            try:
                self.collection.add(
                    documents=[combined_text],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "test_case_id": test_case.id,
                            "title": test_case.title,
                            "priority": test_case.priority.value,
                            "tags": ",".join(test_case.tags),
                            "tags_json": json.dumps(test_case.tags),
                            "created_at": test_case.created_at.isoformat(),
                            "embedding": json.dumps(embedding),
                            "type": "full",
                        }
                    ],
                    ids=[f"test_case_{test_case.id}_full"],
                )
            except Exception as e:
                # On dimension mismatch, recreate collection with cosine and retry once
                if "dimension" in str(e).lower():
                    try:
                        self.client.delete_collection(
                            name=settings.chroma_collection_name
                        )
                    except Exception:
                        pass
                    self.collection = self.client.get_or_create_collection(
                        name=settings.chroma_collection_name,
                        embedding_function=None,
                        metadata={"hnsw:space": "cosine"},
                    )
                    self.collection.add(
                        documents=[combined_text],
                        embeddings=[embedding],
                        metadatas=[
                            {
                                "test_case_id": test_case.id,
                                "title": test_case.title,
                                "priority": test_case.priority.value,
                                "tags": ",".join(test_case.tags),
                                "tags_json": json.dumps(test_case.tags),
                                "created_at": test_case.created_at.isoformat(),
                                "embedding": json.dumps(embedding),
                                "type": "full",
                            }
                        ],
                        ids=[f"test_case_{test_case.id}_full"],
                    )
                else:
                    raise

            # Short doc (mirrors query focus)
            short_text = (
                f"Feature: {test_case.feature_description}\n"
                f"Acceptance: {test_case.acceptance_criteria}\n"
                f"Tags: {', '.join(test_case.tags)}\n"
                f"Priority: {test_case.priority.value}"
            )
            short_embedding = self._get_embedding(short_text)
            if short_embedding:
                try:
                    self.collection.add(
                        documents=[short_text],
                        embeddings=[short_embedding],
                        metadatas=[
                            {
                                "test_case_id": test_case.id,
                                "title": test_case.title,
                                "priority": test_case.priority.value,
                                "tags": ",".join(test_case.tags),
                                "tags_json": json.dumps(test_case.tags),
                                "created_at": test_case.created_at.isoformat(),
                                "embedding": json.dumps(short_embedding),
                                "type": "short",
                            }
                        ],
                        ids=[f"test_case_{test_case.id}_short"],
                    )
                except Exception:
                    pass

            # Store embedding blob in SQLite for local compare
            try:
                db_case = (
                    self.db.query(TestCaseModel)
                    .filter(TestCaseModel.id == test_case.id)
                    .first()
                )
                if db_case:
                    vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    setattr(db_case, "embedding_vector", vec_bytes)
                    self.db.commit()
            except Exception as db_e:
                logger.warning("Failed to store embedding in SQLite", error=str(db_e))
            return True
        except Exception as e:
            logger.error(
                "Failed to store test case",
                test_case_id=getattr(test_case, "id", None),
                error=str(e),
            )
            logger.debug(traceback.format_exc())
            return False

    async def search_similar(
        self,
        feature_description: str,
        limit: int = 5,
        threshold: float = 0.7,
        tags: list[str] | None = None,
        priority: str | None = None,
    ) -> List[SimilarTestCase]:
        """Search similar cases: Chroma first (short docs + filters), then top-up with SQLite."""
        from time import perf_counter

        t0 = perf_counter()
        embedding_query = self._get_embedding(feature_description)
        if not embedding_query:
            return []

        # Chroma query with short-doc filter and optional priority
        chroma_where = {"type": "short"}
        if priority:
            chroma_where["priority"] = priority

        chroma_similar: list[SimilarTestCase] = []
        try:
            chroma_results = self.collection.query(
                query_embeddings=[embedding_query], n_results=limit, where=chroma_where  # type: ignore[arg-type]
            )
            docs = (
                chroma_results.get("documents")
                if isinstance(chroma_results, dict)
                else None
            )
            if not docs or (isinstance(docs, list) and docs and not docs[0]):
                chroma_results = self.collection.query(
                    query_embeddings=[embedding_query], n_results=limit
                )
            docs = (
                chroma_results.get("documents")
                if isinstance(chroma_results, dict)
                else None
            )
            dists = (
                chroma_results.get("distances")
                if isinstance(chroma_results, dict)
                else None
            )
            metas = (
                chroma_results.get("metadatas")
                if isinstance(chroma_results, dict)
                else None
            )
            if docs and dists and metas:
                for doc, distance, meta in zip(docs[0], dists[0], metas[0]):
                    score = 1 - distance
                    if score < threshold:
                        continue
                    # tag filter
                    if tags:
                        try:
                            mj = (
                                meta.get("tags_json")
                                if isinstance(meta, dict)
                                else None
                            )
                            if isinstance(mj, str):
                                arr = json.loads(mj)
                            elif isinstance(mj, list):
                                arr = mj
                            else:
                                arr = []
                            s = {str(x).strip().lower() for x in arr}
                            if not all(str(t).strip().lower() in s for t in tags):
                                continue
                        except Exception:
                            pass
                    tcid = meta.get("test_case_id") if isinstance(meta, dict) else None
                    try:
                        db_id = int(tcid) if tcid is not None else None
                    except Exception:
                        db_id = None
                    if db_id is None:
                        continue
                    db_case = (
                        self.db.query(TestCaseModel)
                        .filter(TestCaseModel.id == db_id)
                        .first()
                    )
                    if db_case:
                        chroma_similar.append(
                            SimilarTestCase(
                                test_case=TestCase.model_validate(db_case),
                                similarity_score=score,
                            )
                        )
        except Exception:
            pass

        if len(chroma_similar) >= limit:
            chroma_similar.sort(key=lambda x: x.similarity_score, reverse=True)
            return chroma_similar[:limit]

        # SQLite top-up: scan recent to bound cost
        SQLITE_SCAN_CAP = 500
        sqlite_similar: list[SimilarTestCase] = []
        try:
            db_cases = (
                self.db.query(TestCaseModel)
                .order_by(TestCaseModel.updated_at.desc())
                .limit(SQLITE_SCAN_CAP)
                .all()
            )
        except Exception:
            db_cases = self.db.query(TestCaseModel).all()

        for db_case in db_cases:
            # quick pre-filters
            if priority:
                try:
                    if str(db_case.priority.value) != str(priority):
                        continue
                except Exception:
                    if str(getattr(db_case, "priority", "")) != str(priority):
                        continue
            if tags and getattr(db_case, "tags", None):
                try:
                    tagset = {str(x).strip().lower() for x in (db_case.tags or [])}
                    if not all(str(t).strip().lower() in tagset for t in tags):
                        continue
                except Exception:
                    pass
            b = getattr(db_case, "embedding_vector", None)
            if not isinstance(b, (bytes, bytearray)) or not b:
                continue
            emb = np.frombuffer(b, dtype=np.float32)
            if self.embedding_dim is not None and len(emb) != len(embedding_query):
                continue
            sim = self._cosine_similarity(embedding_query, emb)
            if sim >= threshold:
                sqlite_similar.append(
                    SimilarTestCase(
                        test_case=TestCase.model_validate(db_case), similarity_score=sim
                    )
                )

        # Merge, prefer higher score
        merged = {sc.test_case.id: sc for sc in chroma_similar}
        for sc in sqlite_similar:
            prev = merged.get(sc.test_case.id)
            if prev is None or sc.similarity_score > prev.similarity_score:
                merged[sc.test_case.id] = sc
        results = list(merged.values())
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]

    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        d = np.dot(vec1, vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        return float(d / (n1 * n2)) if n1 != 0 and n2 != 0 else 0.0

    async def health_check(self) -> dict:
        result = {"chroma": False, "sqlite": False}
        try:
            collections = self.client.list_collections()
            result["chroma"] = any(
                c.name == settings.chroma_collection_name for c in collections
            )
        except Exception:
            pass
        try:
            from sqlalchemy import text

            _ = self.db.execute(text("SELECT 1")).fetchone()
            result["sqlite"] = True
        except Exception:
            pass
        return result

    async def update_test_case_embedding(self, test_case: TestCase) -> bool:
        try:
            await self.delete_test_case_embedding(test_case.id)
            return await self.store_test_case(test_case)
        except Exception:
            return False

    async def delete_test_case_embedding(self, test_case_id: int) -> bool:
        try:
            self.collection.delete(
                ids=[
                    f"test_case_{test_case_id}_full",
                    f"test_case_{test_case_id}_short",
                ]
            )
            db_case = (
                self.db.query(TestCaseModel)
                .filter(TestCaseModel.id == test_case_id)
                .first()
            )
            if db_case:
                setattr(db_case, "embedding_vector", None)
                self.db.commit()
            return True
        except Exception:
            return False
