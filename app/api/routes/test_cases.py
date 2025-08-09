from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import structlog

from app.models.schemas import (
    TestCase, TestCaseCreate, TestCaseUpdate,
    GenerateTestCaseRequest, GenerateTestCaseResponse,
    SearchSimilarRequest, SimilarTestCase
)
from app.services.test_case_service import TestCaseService
from app.core.dependencies import get_test_case_service
from app.core.database import get_database

logger = structlog.get_logger()

router = APIRouter(prefix="/test-cases", tags=["test-cases"])


@router.post("/generate", response_model=GenerateTestCaseResponse)
async def generate_test_case(
    request: GenerateTestCaseRequest,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Generate a new test case using AI and memory search"""
    try:
        logger.info("Generating test case", feature_description=request.feature_description[:100])
        response = await service.generate_test_case(request)
        return response
    except Exception as e:
        logger.error("Failed to generate test case", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate test case"
        )


@router.post("/search-similar", response_model=List[SimilarTestCase])
async def search_similar_test_cases(
    request: SearchSimilarRequest,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Search for similar test cases"""
    try:
        logger.info("Searching similar test cases", feature_description=request.feature_description[:100])
        similar_cases = await service.search_similar_test_cases(request)
        return similar_cases
    except Exception as e:
        logger.error("Failed to search similar test cases", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search similar test cases"
        )


@router.get("/{test_case_id}", response_model=TestCase)
async def get_test_case(
    test_case_id: int,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Get a test case by ID"""
    test_case = await service.get_test_case(test_case_id)
    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    return test_case


@router.get("/", response_model=List[TestCase])
async def get_all_test_cases(
    skip: int = 0,
    limit: int = 100,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Get all test cases with pagination"""
    test_cases = await service.get_all_test_cases(skip=skip, limit=limit)
    return test_cases


@router.put("/{test_case_id}", response_model=TestCase)
async def update_test_case(
    test_case_id: int,
    update_data: TestCaseUpdate,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Update an existing test case"""
    updated_test_case = await service.update_test_case(test_case_id, update_data)
    if not updated_test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    return updated_test_case


@router.delete("/{test_case_id}")
async def delete_test_case(
    test_case_id: int,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Delete a test case"""
    success = await service.delete_test_case(test_case_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    return {"message": "Test case deleted successfully"}


@router.post("/{test_case_id}/improve")
async def improve_test_case(
    test_case_id: int,
    feedback: str,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Improve a test case using AI based on feedback"""
    try:
        improved_test_case = await service.improve_test_case_with_ai(test_case_id, feedback)
        if not improved_test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test case not found"
            )
        return improved_test_case
    except Exception as e:
        logger.error("Failed to improve test case", test_case_id=test_case_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to improve test case"
        )
