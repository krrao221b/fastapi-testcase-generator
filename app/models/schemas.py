from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TestCaseStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class TestCasePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestStep(BaseModel):
    step_number: int = Field(..., description="Step sequence number")
    action: str = Field(..., description="Action to be performed")
    expected_result: str = Field(..., description="Expected result of the action")
    test_data: Optional[str] = Field(None, description="Test data required for this step")


class TestCaseBase(BaseModel):
    title: str = Field(..., description="Test case title")
    description: str = Field(..., description="Detailed description of the test case")
    feature_description: str = Field(..., description="Description of the feature being tested")
    acceptance_criteria: str = Field(..., description="Acceptance criteria for the feature")
    priority: TestCasePriority = Field(default=TestCasePriority.MEDIUM)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    preconditions: Optional[str] = Field(None, description="Preconditions for test execution")
    test_steps: List[TestStep] = Field(..., description="List of test steps")
    expected_result: str = Field(..., description="Overall expected result")


class TestCaseCreate(TestCaseBase):
    pass


class TestCaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[TestCasePriority] = None
    tags: Optional[List[str]] = None
    preconditions: Optional[str] = None
    test_steps: Optional[List[TestStep]] = None
    expected_result: Optional[str] = None
    status: Optional[TestCaseStatus] = None


class TestCase(TestCaseBase):
    id: int
    status: TestCaseStatus = TestCaseStatus.DRAFT
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    jira_issue_key: Optional[str] = None
    zephyr_test_id: Optional[str] = None

    class Config:
        from_attributes = True


class GenerateTestCaseRequest(BaseModel):
    feature_description: str = Field(..., description="Description of the feature to test")
    acceptance_criteria: str = Field(..., description="Acceptance criteria for the feature")
    additional_context: Optional[str] = Field(None, description="Additional context or requirements")
    priority: TestCasePriority = Field(default=TestCasePriority.MEDIUM)
    tags: List[str] = Field(default_factory=list)


class SearchSimilarRequest(BaseModel):
    feature_description: str = Field(..., description="Feature description to search for")
    limit: int = Field(default=5, ge=1, le=20, description="Number of similar test cases to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class SimilarTestCase(BaseModel):
    test_case: TestCase
    similarity_score: float = Field(..., description="Similarity score between 0 and 1")


class GenerateTestCaseResponse(BaseModel):
    test_case: TestCase
    similar_cases: List[SimilarTestCase] = Field(default_factory=list)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
