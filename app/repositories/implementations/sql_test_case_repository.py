from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.repositories.interfaces.test_case_repository import ITestCaseRepository
from app.models.database import TestCaseModel
from app.models.schemas import TestCase, TestCaseCreate, TestCaseUpdate


class SQLTestCaseRepository(ITestCaseRepository):
    """SQLAlchemy implementation of test case repository"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create(self, test_case: TestCaseCreate) -> TestCase:
        """Create a new test case"""
        db_test_case = TestCaseModel(**test_case.model_dump())
        self.db.add(db_test_case)
        self.db.commit()
        self.db.refresh(db_test_case)
        return TestCase.model_validate(db_test_case)
    
    async def get_by_id(self, test_case_id: int) -> Optional[TestCase]:
        """Get test case by ID"""
        db_test_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == test_case_id).first()
        if db_test_case:
            return TestCase.model_validate(db_test_case)
        return None
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[TestCase]:
        """Get all test cases with pagination"""
        db_test_cases = self.db.query(TestCaseModel).offset(skip).limit(limit).all()
        return [TestCase.model_validate(test_case) for test_case in db_test_cases]
    
    async def update(self, test_case_id: int, test_case_update: TestCaseUpdate) -> Optional[TestCase]:
        """Update an existing test case"""
        db_test_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == test_case_id).first()
        if not db_test_case:
            return None
        
        update_data = test_case_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_test_case, field, value)
        
        self.db.commit()
        self.db.refresh(db_test_case)
        return TestCase.model_validate(db_test_case)
    
    async def delete(self, test_case_id: int) -> bool:
        """Delete a test case"""
        db_test_case = self.db.query(TestCaseModel).filter(TestCaseModel.id == test_case_id).first()
        if not db_test_case:
            return False
        
        self.db.delete(db_test_case)
        self.db.commit()
        return True
    
    async def search_by_tags(self, tags: List[str]) -> List[TestCase]:
        """Search test cases by tags"""
        # Note: This is a simplified implementation
        # In production, you might want to use PostgreSQL's array operators or a dedicated search solution
        db_test_cases = self.db.query(TestCaseModel).all()
        matching_cases = []
        
        for test_case in db_test_cases:
            if test_case.tags and any(tag in test_case.tags for tag in tags):
                matching_cases.append(TestCase.model_validate(test_case))
        
        return matching_cases
