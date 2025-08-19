from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from app.models.schemas import GenerateTestCaseRequest, GenerateNewTestCaseRequest, TestCase


class IAIService(ABC):
    """Interface for AI/LLM operations"""
    
    @abstractmethod
    async def generate_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        """Generate a test case using AI"""
        pass

    @abstractmethod
    async def generate_new_test_case(self, request: GenerateNewTestCaseRequest) -> TestCase:
        """Generate a new test case using AI without checking for similar test cases in database"""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text"""
        pass
    
    @abstractmethod
    async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
        """Improve an existing test case based on feedback"""
        pass
