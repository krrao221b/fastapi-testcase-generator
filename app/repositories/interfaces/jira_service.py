from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class IJiraService(ABC):
    """Interface for JIRA integration operations"""
    
    @abstractmethod
    async def create_issue(self, title: str, description: str, issue_type: str = "Story") -> Optional[str]:
        """Create a JIRA issue and return the issue key"""
        pass
    
    @abstractmethod
    async def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Get JIRA issue details"""
        pass
    
    @abstractmethod
    async def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> bool:
        """Update JIRA issue fields"""
        pass
    
    @abstractmethod
    async def add_comment(self, issue_key: str, comment: str) -> bool:
        """Add comment to JIRA issue"""
        pass
    
    @abstractmethod
    async def get_acceptance_criteria(self, issue_key: str) -> Optional[str]:
        """Extract acceptance criteria from JIRA issue"""
        pass
