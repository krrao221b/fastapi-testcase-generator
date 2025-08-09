from typing import List, Optional
import structlog
from app.models.schemas import (
    TestCase, TestCaseCreate, TestCaseUpdate, 
    GenerateTestCaseRequest, GenerateTestCaseResponse, 
    SearchSimilarRequest, SimilarTestCase
)
from app.repositories.interfaces.test_case_repository import ITestCaseRepository
from app.repositories.interfaces.memory_service import IMemoryService
from app.repositories.interfaces.ai_service import IAIService
from app.repositories.interfaces.jira_service import IJiraService
from app.repositories.interfaces.zephyr_service import IZephyrService

logger = structlog.get_logger()


class TestCaseService:
    """Business logic service for test case operations"""
    
    def __init__(
        self,
        test_case_repository: ITestCaseRepository,
        memory_service: IMemoryService,
        ai_service: IAIService,
        jira_service: IJiraService,
        zephyr_service: IZephyrService
    ):
        self.test_case_repository = test_case_repository
        self.memory_service = memory_service
        self.ai_service = ai_service
        self.jira_service = jira_service
        self.zephyr_service = zephyr_service
    
    async def generate_test_case(self, request: GenerateTestCaseRequest) -> GenerateTestCaseResponse:
        """Generate a new test case using AI and memory search"""
        try:
            logger.info("Starting test case generation", 
                       feature_description=request.feature_description[:100])
            
            # First, search for similar test cases
            similar_cases = await self.memory_service.search_similar(
                feature_description=request.feature_description,
                limit=5,
                threshold=0.7
            )
            
            # Generate new test case using AI
            ai_test_case = await self.ai_service.generate_test_case(request)
            
            # Save the generated test case to database
            test_case_create = TestCaseCreate(**ai_test_case.dict(exclude={"id", "created_at", "updated_at"}))
            saved_test_case = await self.test_case_repository.create(test_case_create)
            
            # Store in vector database for future similarity searches
            await self.memory_service.store_test_case(saved_test_case)
            
            logger.info("Test case generated and saved successfully", 
                       test_case_id=saved_test_case.id)
            
            return GenerateTestCaseResponse(
                test_case=saved_test_case,
                similar_cases=similar_cases,
                generation_metadata={
                    "ai_model_used": "openai",
                    "similar_cases_found": len(similar_cases),
                    "generation_timestamp": saved_test_case.created_at.isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to generate test case", 
                        feature_description=request.feature_description, error=str(e))
            raise
    
    async def search_similar_test_cases(self, request: SearchSimilarRequest) -> List[SimilarTestCase]:
        """Search for similar test cases"""
        try:
            similar_cases = await self.memory_service.search_similar(
                feature_description=request.feature_description,
                limit=request.limit,
                threshold=request.similarity_threshold
            )
            
            logger.info("Similar test cases search completed", 
                       feature_description=request.feature_description[:100],
                       results_count=len(similar_cases))
            
            return similar_cases
            
        except Exception as e:
            logger.error("Failed to search similar test cases", 
                        feature_description=request.feature_description, error=str(e))
            raise
    
    async def get_test_case(self, test_case_id: int) -> Optional[TestCase]:
        """Get a test case by ID"""
        return await self.test_case_repository.get_by_id(test_case_id)
    
    async def get_all_test_cases(self, skip: int = 0, limit: int = 100) -> List[TestCase]:
        """Get all test cases with pagination"""
        return await self.test_case_repository.get_all(skip=skip, limit=limit)
    
    async def update_test_case(self, test_case_id: int, update_data: TestCaseUpdate) -> Optional[TestCase]:
        """Update an existing test case"""
        try:
            updated_test_case = await self.test_case_repository.update(test_case_id, update_data)
            
            if updated_test_case:
                # Update the vector database embedding
                await self.memory_service.update_test_case_embedding(updated_test_case)
                
                logger.info("Test case updated successfully", test_case_id=test_case_id)
            
            return updated_test_case
            
        except Exception as e:
            logger.error("Failed to update test case", test_case_id=test_case_id, error=str(e))
            raise
    
    async def delete_test_case(self, test_case_id: int) -> bool:
        """Delete a test case"""
        try:
            # Delete from vector database first
            await self.memory_service.delete_test_case_embedding(test_case_id)
            
            # Delete from main database
            success = await self.test_case_repository.delete(test_case_id)
            
            if success:
                logger.info("Test case deleted successfully", test_case_id=test_case_id)
            
            return success
            
        except Exception as e:
            logger.error("Failed to delete test case", test_case_id=test_case_id, error=str(e))
            raise
    
    async def integrate_with_jira(self, test_case_id: int, create_issue: bool = False) -> Optional[str]:
        """Integrate test case with JIRA"""
        try:
            test_case = await self.test_case_repository.get_by_id(test_case_id)
            if not test_case:
                return None
            
            if create_issue:
                # Create JIRA issue
                issue_key = await self.jira_service.create_issue(
                    title=test_case.title,
                    description=f"{test_case.description}\n\nFeature: {test_case.feature_description}\n\nAcceptance Criteria: {test_case.acceptance_criteria}"
                )
                
                if issue_key:
                    # Update test case with JIRA issue key
                    await self.test_case_repository.update(
                        test_case_id, 
                        TestCaseUpdate(jira_issue_key=issue_key)
                    )
                    
                    logger.info("Test case integrated with JIRA", 
                              test_case_id=test_case_id, issue_key=issue_key)
                
                return issue_key
            
            return test_case.jira_issue_key
            
        except Exception as e:
            logger.error("Failed to integrate with JIRA", 
                        test_case_id=test_case_id, error=str(e))
            raise
    
    async def integrate_with_zephyr(self, test_case_id: int, project_key: str) -> Optional[str]:
        """Integrate test case with Zephyr"""
        try:
            test_case = await self.test_case_repository.get_by_id(test_case_id)
            if not test_case:
                return None
            
            # Create test case in Zephyr
            zephyr_test_id = await self.zephyr_service.create_test_case(test_case, project_key)
            
            if zephyr_test_id:
                # Update test case with Zephyr test ID
                await self.test_case_repository.update(
                    test_case_id,
                    TestCaseUpdate(zephyr_test_id=zephyr_test_id)
                )
                
                logger.info("Test case integrated with Zephyr", 
                          test_case_id=test_case_id, zephyr_test_id=zephyr_test_id)
            
            return zephyr_test_id
            
        except Exception as e:
            logger.error("Failed to integrate with Zephyr", 
                        test_case_id=test_case_id, error=str(e))
            raise
    
    async def improve_test_case_with_ai(self, test_case_id: int, feedback: str) -> Optional[TestCase]:
        """Improve a test case using AI based on feedback"""
        try:
            test_case = await self.test_case_repository.get_by_id(test_case_id)
            if not test_case:
                return None
            
            # Use AI to improve the test case
            improved_test_case = await self.ai_service.improve_test_case(test_case, feedback)
            
            # Update the test case in database
            update_data = TestCaseUpdate(
                title=improved_test_case.title,
                description=improved_test_case.description,
                test_steps=improved_test_case.test_steps,
                expected_result=improved_test_case.expected_result,
                preconditions=improved_test_case.preconditions
            )
            
            updated_test_case = await self.test_case_repository.update(test_case_id, update_data)
            
            if updated_test_case:
                # Update vector database
                await self.memory_service.update_test_case_embedding(updated_test_case)
                
                logger.info("Test case improved with AI", test_case_id=test_case_id)
            
            return updated_test_case
            
        except Exception as e:
            logger.error("Failed to improve test case with AI", 
                        test_case_id=test_case_id, error=str(e))
            raise
