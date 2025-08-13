from fastapi import APIRouter, Depends, HTTPException, status
import structlog

from app.services.test_case_service import TestCaseService
from app.core.dependencies import get_test_case_service

logger = structlog.get_logger()

router = APIRouter(prefix="/integrations", tags=["integrations"])



# GET endpoint to fetch JIRA ticket details by ticket key
@router.get("/jira/{ticket_key}")
async def get_jira_ticket(
    ticket_key: str,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Fetch JIRA ticket details by ticket key (e.g., PROJ-123)"""
    try:
        logger.info("Fetching JIRA ticket", ticket_key=ticket_key)
        # Replace with actual service call to fetch ticket details
        ticket_data = await service.get_jira_ticket(ticket_key)
        if not ticket_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"JIRA ticket {ticket_key} not found"
            )
        return ticket_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch JIRA ticket", ticket_key=ticket_key, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch JIRA ticket {ticket_key}"
        )

# Existing POST endpoint for JIRA integration
@router.post("/jira/{test_case_id}")
async def integrate_with_jira(
    test_case_id: int,
    create_issue: bool = False,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Integrate test case with JIRA"""
    try:
        issue_key = await service.integrate_with_jira(test_case_id, create_issue)
        if not issue_key:
            if create_issue:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create JIRA issue"
                )
            else:
                return {"message": "No JIRA integration found for this test case"}
        return {
            "test_case_id": test_case_id,
            "jira_issue_key": issue_key,
            "action": "created" if create_issue else "retrieved"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to integrate with JIRA", test_case_id=test_case_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to integrate with JIRA"
        )


@router.post("/zephyr/{test_case_id}")
async def integrate_with_zephyr(
    test_case_id: int,
    project_key: str,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Integrate test case with Zephyr"""
    try:
        zephyr_test_id = await service.integrate_with_zephyr(test_case_id, project_key)
        if not zephyr_test_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create Zephyr test case"
            )
        
        return {
            "test_case_id": test_case_id,
            "zephyr_test_id": zephyr_test_id,
            "project_key": project_key
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to integrate with Zephyr", test_case_id=test_case_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to integrate with Zephyr"
        )
