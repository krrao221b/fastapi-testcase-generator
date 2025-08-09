import chromadb
from typing import List
from app.repositories.interfaces.memory_service import IMemoryService
from app.models.schemas import TestCase, SimilarTestCase
from app.config.settings import settings
import structlog

logger = structlog.get_logger()


class ChromaMemoryService(IMemoryService):
    """ChromaDB implementation of memory service"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    async def store_test_case(self, test_case: TestCase) -> bool:
        """Store a test case in the vector database"""
        try:
            # Create a combined text for embedding
            combined_text = f"{test_case.title} {test_case.description} {test_case.feature_description} {test_case.acceptance_criteria}"
            
            # Store in ChromaDB
            self.collection.add(
                documents=[combined_text],
                metadatas=[{
                    "test_case_id": test_case.id,
                    "title": test_case.title,
                    "priority": test_case.priority.value,
                    "tags": ",".join(test_case.tags),
                    "created_at": test_case.created_at.isoformat()
                }],
                ids=[f"test_case_{test_case.id}"]
            )
            
            logger.info("Test case stored in vector database", test_case_id=test_case.id)
            return True
            
        except Exception as e:
            logger.error("Failed to store test case in vector database", 
                        test_case_id=test_case.id, error=str(e))
            return False
    
    async def search_similar(self, feature_description: str, limit: int = 5, threshold: float = 0.7) -> List[SimilarTestCase]:
        """Search for similar test cases based on feature description"""
        try:
            results = self.collection.query(
                query_texts=[feature_description],
                n_results=limit
            )
            
            similar_cases = []
            if results['documents'] and results['distances'] and results['metadatas']:
                for i, (doc, distance, metadata) in enumerate(zip(
                    results['documents'][0], 
                    results['distances'][0], 
                    results['metadatas'][0]
                )):
                    # Convert distance to similarity score (1 - cosine_distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= threshold:
                        # Note: In a real implementation, you'd fetch the full TestCase from the database
                        # This is a simplified version
                        test_case_data = {
                            "id": metadata["test_case_id"],
                            "title": metadata["title"],
                            "description": doc,  # This is simplified
                            "feature_description": feature_description,
                            "acceptance_criteria": "",
                            "priority": metadata["priority"],
                            "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                            "preconditions": "",
                            "test_steps": [],
                            "expected_result": "",
                            "status": "active",
                            "created_at": metadata["created_at"],
                            "updated_at": metadata["created_at"]
                        }
                        
                        similar_case = SimilarTestCase(
                            test_case=TestCase(**test_case_data),
                            similarity_score=similarity_score
                        )
                        similar_cases.append(similar_case)
            
            logger.info("Similar test cases found", 
                       feature_description=feature_description, 
                       count=len(similar_cases))
            return similar_cases
            
        except Exception as e:
            logger.error("Failed to search similar test cases", 
                        feature_description=feature_description, error=str(e))
            return []
    
    async def update_test_case_embedding(self, test_case: TestCase) -> bool:
        """Update the embedding for an existing test case"""
        try:
            # Delete old embedding
            await self.delete_test_case_embedding(test_case.id)
            
            # Store new embedding
            return await self.store_test_case(test_case)
            
        except Exception as e:
            logger.error("Failed to update test case embedding", 
                        test_case_id=test_case.id, error=str(e))
            return False
    
    async def delete_test_case_embedding(self, test_case_id: int) -> bool:
        """Delete the embedding for a test case"""
        try:
            self.collection.delete(ids=[f"test_case_{test_case_id}"])
            logger.info("Test case embedding deleted", test_case_id=test_case_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete test case embedding", 
                        test_case_id=test_case_id, error=str(e))
            return False
