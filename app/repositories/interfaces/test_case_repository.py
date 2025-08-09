from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.schemas import TestCase, TestCaseCreate, TestCaseUpdate


class ITestCaseRepository(ABC):
    """Interface for test case repository operations"""
    
    @abstractmethod
    async def create(self, test_case: TestCaseCreate) -> TestCase:
        pass
    
    @abstractmethod
    async def get_by_id(self, test_case_id: int) -> Optional[TestCase]:
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[TestCase]:
        pass
    
    @abstractmethod
    async def update(self, test_case_id: int, test_case_update: TestCaseUpdate) -> Optional[TestCase]:
        pass
    
    @abstractmethod
    async def delete(self, test_case_id: int) -> bool:
        pass
    
    @abstractmethod
    async def search_by_tags(self, tags: List[str]) -> List[TestCase]:
        pass
