from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from app.models.schemas import TestCase


class IZephyrService(ABC):
    """Interface for Zephyr integration operations"""
    
    @abstractmethod
    async def create_test_case(self, test_case: TestCase, project_key: str) -> Optional[str]:
        """Create a test case in Zephyr and return the test ID"""
        pass
    
    @abstractmethod
    async def update_test_case(self, zephyr_test_id: str, test_case: TestCase) -> bool:
        """Update an existing test case in Zephyr"""
        pass
    
    @abstractmethod
    async def execute_test(self, zephyr_test_id: str, execution_status: str, comment: Optional[str] = None) -> bool:
        """Execute a test case in Zephyr with given status"""
        pass
    
    @abstractmethod
    async def get_test_execution_history(self, zephyr_test_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a test case"""
        pass
    
    @abstractmethod
    async def create_test_cycle(self, name: str, project_key: str, version_id: str) -> Optional[str]:
        """Create a test cycle in Zephyr"""
        pass
