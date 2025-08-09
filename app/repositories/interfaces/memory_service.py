from abc import ABC, abstractmethod
from typing import List
from app.models.schemas import TestCase, SimilarTestCase


class IMemoryService(ABC):
    """Interface for memory/vector database operations"""
    
    @abstractmethod
    async def store_test_case(self, test_case: TestCase) -> bool:
        """Store a test case in the vector database"""
        pass
    
    @abstractmethod
    async def search_similar(self, feature_description: str, limit: int = 5, threshold: float = 0.7) -> List[SimilarTestCase]:
        """Search for similar test cases based on feature description"""
        pass
    
    @abstractmethod
    async def update_test_case_embedding(self, test_case: TestCase) -> bool:
        """Update the embedding for an existing test case"""
        pass
    
    @abstractmethod
    async def delete_test_case_embedding(self, test_case_id: int) -> bool:
        """Delete the embedding for a test case"""
        pass
