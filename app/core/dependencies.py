from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session
from app.repositories.interfaces.test_case_repository import ITestCaseRepository
from app.repositories.interfaces.memory_service import IMemoryService
from app.repositories.interfaces.ai_service import IAIService
from app.repositories.interfaces.jira_service import IJiraService
from app.repositories.interfaces.zephyr_service import IZephyrService

from app.repositories.implementations.sql_test_case_repository import SQLTestCaseRepository
from app.repositories.implementations.chroma_memory_service import ChromaMemoryService
from app.repositories.implementations.gemini_service import GeminiService
from app.repositories.implementations.jira_service import AtlassianJiraService
from app.repositories.implementations.zephyr_service import ZephyrScaleService

from app.services.test_case_service import TestCaseService
from app.core.database import get_database


class Container:
    """Dependency injection container"""
    
    def __init__(self):
        self._test_case_repository = None
        self._memory_service = None
        self._ai_service = None
        self._jira_service = None
        self._zephyr_service = None
        self._test_case_service = None
    
    def test_case_repository(self, db: Session) -> ITestCaseRepository:
        """Get test case repository instance"""
        return SQLTestCaseRepository(db)
    
    @lru_cache()
    def memory_service(self) -> IMemoryService:
        """Get memory service instance (singleton)"""
        if self._memory_service is None:
            self._memory_service = ChromaMemoryService()
        return self._memory_service
    
    @lru_cache()
    def ai_service(self) -> IAIService:
        """Get AI service instance (singleton)"""
        if self._ai_service is None:
            self._ai_service = GeminiService()
        return self._ai_service
    
    @lru_cache()
    def jira_service(self) -> IJiraService:
        """Get JIRA service instance (singleton)"""
        if self._jira_service is None:
            self._jira_service = AtlassianJiraService()
        return self._jira_service
    
    @lru_cache()
    def zephyr_service(self) -> IZephyrService:
        """Get Zephyr service instance (singleton)"""
        if self._zephyr_service is None:
            self._zephyr_service = ZephyrScaleService()
        return self._zephyr_service
    
    def test_case_service(self, db: Session) -> TestCaseService:
        """Get test case service instance"""
        return TestCaseService(
            test_case_repository=self.test_case_repository(db),
            memory_service=self.memory_service(),
            ai_service=self.ai_service(),
            jira_service=self.jira_service(),
            zephyr_service=self.zephyr_service()
        )


# Global container instance
container = Container()


# Dependency providers for FastAPI
def get_test_case_repository(db: Session = Depends(get_database)) -> ITestCaseRepository:
    """FastAPI dependency for test case repository"""
    return container.test_case_repository(db)


def get_memory_service() -> IMemoryService:
    """FastAPI dependency for memory service"""
    return container.memory_service()


def get_ai_service() -> IAIService:
    """FastAPI dependency for AI service"""
    return container.ai_service()


def get_jira_service() -> IJiraService:
    """FastAPI dependency for JIRA service"""
    return container.jira_service()


def get_zephyr_service() -> IZephyrService:
    """FastAPI dependency for Zephyr service"""
    return container.zephyr_service()


def get_test_case_service(db: Session = Depends(get_database)) -> TestCaseService:
    """FastAPI dependency for test case service"""
    return container.test_case_service(db)
