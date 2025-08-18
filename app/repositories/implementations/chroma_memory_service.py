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
import uuid
from openai import OpenAI
from chromadb.utils import embedding_functions
import traceback
# from sentence_transformers import SentenceTransformer


logger = structlog.get_logger()

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key
)

# Define embedding model
embedding_model = settings.openai_embedding_model


class ChromaMemoryService(IMemoryService):
    """ChromaDB implementation of memory service with embedding and SQLite sync"""

    def __init__(self, db_session: Session):
        logger.debug("Initializing ChromaMemoryService")
        logger.info("Initializing ChromaMemoryService", chroma_path=settings.chroma_persist_directory, collection=settings.chroma_collection_name)
        try:
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
            # Prefer local embeddings if configured. Import sentence-transformers lazily
            self.local_model = None
            if settings.use_local_embeddings:
                logger.info("Using local embedding model", model=settings.local_embedding_model)
                try:
                    # import lazily to avoid hard dependency at startup
                    from sentence_transformers import SentenceTransformer

                    self.local_model = SentenceTransformer(settings.local_embedding_model)
                except ImportError as ie:
                    logger.warning("sentence-transformers is not installed or incompatible", error=str(ie))
                    logger.info("Install with: pip install sentence-transformers")
                    self.local_model = None
                except Exception as e:
                    logger.warning("Failed to load local model", model=settings.local_embedding_model, error=str(e))
                    logger.debug(traceback.format_exc())
                    self.local_model = None

            if self.local_model is None:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.openai_api_key,
                    api_base=settings.openai_base_url,
                    model_name=embedding_model
                )
            else:
                self.embedding_function = None
            # Create or get collection. After creation, verify embedding dimension matches current model
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function
            )

            # Determine expected embedding dimension from the configured model (or local model)
            try:
                self.embedding_dim = self._determine_embedding_dim()
                logger.info("Determined embedding dimension", embedding_dim=self.embedding_dim, model=embedding_model)
            except Exception as ed:
                self.embedding_dim = None
                logger.warning("Unable to determine embedding dimension at startup", error=str(ed))

            # If the collection already has stored embeddings, try to detect a mismatch and recreate if necessary
            try:
                # collection.get() returns dict with 'metadatas' if any docs exist
                existing = self.collection.get()
                metas = existing.get('metadatas') if isinstance(existing, dict) else None
                # metas is a list-of-lists like [[{...}, ...]] or [] depending on Chroma version
                flat_meta = []
                if metas:
                    # handle both nested and flat
                    if isinstance(metas[0], list):
                        for inner in metas:
                            flat_meta.extend(inner)
                    else:
                        flat_meta = metas

                if flat_meta and self.embedding_dim is not None:
                    # if any metadata includes our stored embedding snapshot, compare length
                    for md in flat_meta:
                        emb_json = md.get('embedding') if isinstance(md, dict) else None
                        if emb_json:
                            try:
                                prev_emb = json.loads(emb_json)
                                if isinstance(prev_emb, list) and len(prev_emb) != self.embedding_dim:
                                    logger.warning("Stored embeddings dimension mismatch detected; recreating collection", stored_dim=len(prev_emb), expected_dim=self.embedding_dim)
                                    try:
                                        self.client.delete_collection(name=settings.chroma_collection_name)
                                    except Exception as de:
                                        logger.warning("Failed to delete existing collection during startup check", error=str(de))
                                    # recreate collection with current embedding function
                                    self.collection = self.client.get_or_create_collection(
                                        name=settings.chroma_collection_name,
                                        embedding_function=self.embedding_function
                                    )
                                    break
                            except Exception:
                                # skip malformed embedding metadata
                                continue
            except Exception as e:
                # best-effort; don't fail startup if Chroma introspection fails
                logger.debug("Could not introspect existing collection embeddings at startup", error=str(e))
            self.db = db_session
            logger.debug("ChromaMemoryService initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize ChromaMemoryService", error=str(e))
            logger.debug(traceback.format_exc())
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using openai client (sync)"""
        try:
            # Debug: show short preview of text
            logger.debug("_get_embedding called", text_preview=(text or "")[:200].replace('\n', ' '))
            try:
                api_key = getattr(openai_client, 'api_key', None)
                masked = (api_key[:4] + '...' + api_key[-4:]) if api_key else None
            except Exception:
                masked = None
            logger.info("Calling OpenAI embeddings", model=embedding_model, api_key_masked=masked)

            # If local model available, use it
            if self.local_model is not None:
                emb = self.local_model.encode(text).tolist()
                logger.debug("Local model embedding generated", length=len(emb))
                return emb

            response = openai_client.embeddings.create(
                model=embedding_model,
                input=[text]
            )
            emb = response.data[0].embedding
            logger.debug("Received embedding", length=len(emb))
            return emb
        except Exception as e:
            logger.error("Embedding API error", error=str(e))
            logger.debug(traceback.format_exc())
            return []

    def _determine_embedding_dim(self) -> int:
        """Determine the embedding vector dimension for the configured embedding model.

        If a local model is in use, infer from its output. Otherwise, request a single embedding
        (quietly) to determine dimension. This is a best-effort method and should not raise on failure.
        """
        # If local model available, use a small sample
        if self.local_model is not None:
            try:
                emb = self.local_model.encode("test")
                return len(emb)
            except Exception as e:
                logger.debug("Failed to infer dim from local model", error=str(e))

        # Otherwise, call the remote embeddings API with a tiny prompt
        try:
            resp = openai_client.embeddings.create(model=embedding_model, input=["test"])
            emb = resp.data[0].embedding
            return len(emb)
        except Exception as e:
            logger.debug("Failed to infer dim from OpenAI embedding API", error=str(e))
            # As a fallback, map known models to dims (extendable)
            model_dim_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                # older OpenAI models - keep for compatibility
                "text-embedding-ada-002": 1536,
            }
            return model_dim_map.get(embedding_model, None)
    

    async def store_test_case(self, test_case: TestCase) -> bool:
        """Store a test case in ChromaDB and SQLite with embedding"""
        try:
            combined_text = f"Title: {test_case.title}\nDescription: {test_case.description}\nFeature: {test_case.feature_description}\nAcceptance: {test_case.acceptance_criteria}\nTags: {','.join(test_case.tags)}\nPriority: {test_case.priority.value}"
            embedding = self._get_embedding(combined_text)
            if not embedding:
                raise Exception("Embedding generation failed")

            # Store in ChromaDB: add full document (title+description+feature+acceptance+tags+priority)
            try:
                self.collection.add(
                    documents=[combined_text],
                    metadatas=[{
                        "test_case_id": test_case.id,
                        "title": test_case.title,
                        "priority": test_case.priority.value,
                        "tags": ",".join(test_case.tags),
                        "created_at": test_case.created_at.isoformat(),
                        # keep embedding in metadata for easy migration/debug
                        "embedding": json.dumps(embedding),
                        "type": "full"
                    }],
                    ids=[f"test_case_{test_case.id}_full"]
                )

                # Also add a short-form document that mirrors the search query used by the service
                short_text = f"Feature: {test_case.feature_description}\nAcceptance: {test_case.acceptance_criteria}\nTags: {','.join(test_case.tags)}\nPriority: {test_case.priority.value}"
                short_embedding = self._get_embedding(short_text)
                if short_embedding:
                    try:
                        self.collection.add(
                            documents=[short_text],
                            metadatas=[{
                                "test_case_id": test_case.id,
                                "title": test_case.title,
                                "priority": test_case.priority.value,
                                "tags": ",".join(test_case.tags),
                                "created_at": test_case.created_at.isoformat(),
                                "embedding": json.dumps(short_embedding),
                                "type": "short"
                            }],
                            ids=[f"test_case_{test_case.id}_short"]
                        )
                    except Exception as se:
                        logger.warning("Failed to add short-form doc to Chroma", test_case_id=test_case.id, error=str(se))
                else:
                    logger.warning("Short-form embedding generation failed", test_case_id=test_case.id)
            except Exception as e:
                # Handle dimension mismatch between collection and current embedding
                msg = str(e)
                logger.error("Chroma add failed", error=msg)
                # If it's a dimension mismatch, attempt to recreate collection with current embedding function
                if "Collection expecting embedding with dimension" in msg:
                    logger.warning("Embedding dimension mismatch detected. Recreating Chroma collection to match current embedding dims.")
                    logger.warning("Recreating Chroma collection due to embedding dim mismatch", error=msg)
                    try:
                        self.client.delete_collection(name=settings.chroma_collection_name)
                    except Exception as de:
                        logger.warning("Failed to delete existing collection", error=str(de))
                    # Recreate collection with current embedding function
                    try:
                        self.collection = self.client.get_or_create_collection(
                            name=settings.chroma_collection_name,
                            embedding_function=self.embedding_function
                        )
                        # Retry add once
                        self.collection.add(
                            documents=[combined_text],
                            metadatas=[{
                                "test_case_id": test_case.id,
                                "title": test_case.title,
                                "priority": test_case.priority.value,
                                "tags": ",".join(test_case.tags),
                                "created_at": test_case.created_at.isoformat(),
                                "embedding": json.dumps(embedding)
                            }],
                            ids=[f"test_case_{test_case.id}"]
                        )
                    except Exception as retry_e:
                        logger.error("Retry add after recreating collection failed", error=str(retry_e))
                        logger.debug(traceback.format_exc())
                        raise
                else:
                    raise

            # Store embedding in SQLite as BLOB
            db_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == test_case.id).first()
            if db_case:
                try:
                    db_case.embedding_vector = np.array(embedding, dtype=np.float32).tobytes()
                    self.db.commit()
                    logger.debug("Stored embedding in SQLite", test_case_id=test_case.id)
                except Exception as db_e:
                    logger.error("Failed to store embedding in SQLite", test_case_id=test_case.id, error=str(db_e))
                    logger.debug(traceback.format_exc())

            logger.info("Test case stored in vector DB and SQLite", test_case_id=test_case.id)
            return True
        except Exception as e:
            logger.error("Failed to store test case", test_case_id=getattr(test_case, 'id', None), error=str(e))
            logger.debug(traceback.format_exc())
            return False
    

    async def search_similar(self, feature_description: str, limit: int = 5, threshold: float = 0.7, tags: List[str] = None, priority: str = None) -> List[SimilarTestCase]:
        """Search for similar test cases using all fields and fetch full details from SQLite and ChromaDB"""
        try:
            logger.debug("search_similar called", feature_description_preview=(feature_description or "")[:200].replace('\n', ' '))
            logger.info("search_similar called", feature_description_preview=(feature_description or "")[:200], limit=limit, threshold=threshold)

            embedding_query = self._get_embedding(feature_description)
            if not embedding_query:
                logger.warning("Query embedding failed or empty")
                raise Exception("Query embedding failed")

            # Search in SQLite (local vector compare)
            db_cases = self.db.query(TestCaseModel).all()
            sqlite_results = []
            for db_case in db_cases:
                try:
                    if db_case.embedding_vector:
                        embedding = np.frombuffer(db_case.embedding_vector, dtype=np.float32)
                        # If stored embedding length doesn't match the current query embedding length, skip and log
                        if self.embedding_dim is not None and len(embedding) != len(embedding_query):
                            logger.warning(
                                "SQLite embedding dimension mismatch - skipping DB compare",
                                file="chroma_memory_service.py",
                                method="search_similar",
                                test_case_id=db_case.id,
                                stored_dim=len(embedding),
                                query_dim=len(embedding_query),
                            )
                            continue
                        similarity = self._cosine_similarity(embedding_query, embedding)
                        # Always log per-row similarity for debugging with DB metadata
                        logger.debug(
                            "SQLite per-row similarity",
                            file="chroma_memory_service.py",
                            method="search_similar",
                            test_case_id=db_case.id,
                            title=getattr(db_case, 'title', None),
                            priority=getattr(db_case, 'priority', None),
                            tags=getattr(db_case, 'tags', None),
                            similarity=similarity,
                            threshold=threshold,
                        )
                        if similarity >= threshold:
                            test_case = TestCase.model_validate(db_case)
                            sqlite_results.append(SimilarTestCase(test_case=test_case, similarity_score=similarity))
                except Exception as inner_e:
                    print(f"[error] Error comparing with SQLite embedding for id={getattr(db_case, 'id', None)}: {inner_e}")
                    print(traceback.format_exc())

            # Debug: report sqlite matches with file/method context
            if sqlite_results:
                ids_scores = [(s.test_case.id, round(s.similarity_score, 4)) for s in sqlite_results]
                logger.debug("SQLite matches found", file="chroma_memory_service.py", method="search_similar", matches=ids_scores)
                logger.info("SQLite similarity matches", file="chroma_memory_service.py", method="search_similar", matches=ids_scores)

            # Sort and limit
            sqlite_results.sort(key=lambda x: x.similarity_score, reverse=True)
            sqlite_results = sqlite_results[:limit]

            # Search in ChromaDB
            chroma_results = self.collection.query(
                query_texts=[feature_description],
                n_results=limit
            )
            chroma_similar_cases = []
            try:
                docs = chroma_results.get('documents') if isinstance(chroma_results, dict) else None
                dists = chroma_results.get('distances') if isinstance(chroma_results, dict) else None
                metas = chroma_results.get('metadatas') if isinstance(chroma_results, dict) else None
            except Exception:
                docs = dists = metas = None

            if docs and dists and metas:
                for i, (doc, distance, metadata) in enumerate(zip(docs[0], dists[0], metas[0])):
                    try:
                        similarity_score = 1 - distance
                        # Log chroma per-result similarity for debugging with metadata
                        logger.debug(
                            "Chroma per-result similarity",
                            file="chroma_memory_service.py",
                            method="search_similar",
                            index=i,
                            test_case_id=metadata.get('test_case_id'),
                            title=metadata.get('title'),
                            priority=metadata.get('priority'),
                            tags=metadata.get('tags'),
                            similarity=similarity_score,
                            threshold=threshold,
                        )
                        if similarity_score >= threshold:
                            db_id = int(metadata.get("test_case_id"))
                            db_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == db_id).first()
                            if db_case:
                                test_case = TestCase.model_validate(db_case)
                                chroma_similar_cases.append(SimilarTestCase(test_case=test_case, similarity_score=similarity_score))
                    except Exception as inner_e:
                        print(f"[error] Error processing chroma result idx={i}: {inner_e}")
                        print(traceback.format_exc())

            # Debug: report chroma matches with file/method context
            if chroma_similar_cases:
                ids_scores = [(s.test_case.id, round(s.similarity_score, 4), s.test_case.title) for s in chroma_similar_cases]
                logger.debug("Chroma matches found", file="chroma_memory_service.py", method="search_similar", matches=ids_scores)
                logger.info("Chroma similarity matches", file="chroma_memory_service.py", method="search_similar", matches=ids_scores)

            # Combine results and deduplicate by test_case.id
            all_cases = {sc.test_case.id: sc for sc in sqlite_results}
            for sc in chroma_similar_cases:
                if sc.test_case.id not in all_cases:
                    all_cases[sc.test_case.id] = sc

            final_results = list(all_cases.values())
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = final_results[:limit]

            # Debug: final combined results
            if final_results:
                final_ids_scores = [
                    {
                        "id": s.test_case.id,
                        "title": s.test_case.title,
                        "priority": getattr(s.test_case, 'priority', None),
                        "tags": getattr(s.test_case, 'tags', None),
                        "similarity": round(s.similarity_score, 4),
                    }
                    for s in final_results
                ]
                logger.debug("Final combined results with DB data", file="chroma_memory_service.py", method="search_similar", results=final_ids_scores)
                logger.info("Similar test cases found", file="chroma_memory_service.py", method="search_similar", query=feature_description, count=len(final_results), results=final_ids_scores)
            else:
                logger.debug("No similar test cases found", file="chroma_memory_service.py", method="search_similar", query_preview=(feature_description or '')[:200])
                logger.info("No similar test cases found", file="chroma_memory_service.py", method="search_similar", query_preview=(feature_description or '')[:200])
            logger.debug("search_similar returning results_count", count=len(final_results))
            return final_results
        except Exception as e:
            print(f"[error] Failed to search similar test cases: {e}")
            print(traceback.format_exc())
            logger.error("Failed to search similar test cases", query=feature_description, error=str(e))
            return []

    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        # Defensive: if dims mismatch, log and return similarity 0 rather than raising
        if vec1.shape != vec2.shape:
            logger.warning(
                "Cosine similarity skipped due to shape mismatch",
                file="chroma_memory_service.py",
                method="_cosine_similarity",
                shape_vec1=vec1.shape,
                shape_vec2=vec2.shape,
            )
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
    

    async def health_check(self) -> dict:
        """Simple health check for the memory service"""
        result = {"chroma": False, "sqlite": False}
        try:
            # Check Chroma collection exists
            collections = self.client.list_collections()
            result["chroma"] = any(c.name == settings.chroma_collection_name for c in collections)
        except Exception as e:
            print(f"[warn] Chroma health check failed: {e}")
            print(traceback.format_exc())

        try:
            # Quick sqlite query
            _ = self.db.execute("SELECT 1").fetchone()
            result["sqlite"] = True
        except Exception as e:
            print(f"[warn] SQLite health check failed: {e}")
            print(traceback.format_exc())

        return result

    async def update_test_case_embedding(self, test_case: TestCase) -> bool:
        """Update the embedding for an existing test case in both DBs"""
        try:
            await self.delete_test_case_embedding(test_case.id)
            return await self.store_test_case(test_case)
        except Exception as e:
            logger.error("Failed to update test case embedding", test_case_id=test_case.id, error=str(e))
            return False
    

    async def delete_test_case_embedding(self, test_case_id: int) -> bool:
        """Delete the embedding for a test case in both DBs"""
        try:
            self.collection.delete(ids=[f"test_case_{test_case_id}"])
            db_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == test_case_id).first()
            if db_case:
                db_case.embedding_vector = None
                self.db.commit()
            logger.info("Test case embedding deleted", test_case_id=test_case_id)
            return True
        except Exception as e:
            logger.error("Failed to delete test case embedding", test_case_id=test_case_id, error=str(e))
            return False
