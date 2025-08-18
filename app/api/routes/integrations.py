import structlog
import httpx

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.zephyr import PushTestcaseRequest, PushTestcaseResponse
from app.services.zephyr_service import ZephyrService, JiraService    
from app.services.test_case_service import TestCaseService
from app.core.dependencies import get_test_case_service

logger = structlog.get_logger()

router = APIRouter(prefix="/integrations", tags=["integrations"])



# GET endpoint to fetch JIRA ticket details by ticket key
@router.get("/jira/{ticket_key}")
async def get_jira_ticket(
    ticket_key: str,
    include_similar: bool = True,
    limit: int = 5,
    threshold: float = 0.7,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Fetch JIRA ticket details by ticket key (e.g., PROJ-123)"""
    try:
        logger.info("Fetching JIRA ticket", ticket_key=ticket_key)
        # Replace with actual service call to fetch ticket details
        ticket_data = await service.get_jira_ticket(ticket_key, include_similar=include_similar, limit=limit, threshold=threshold)
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

@router.post("/zephyr/push", response_model=PushTestcaseResponse, status_code=status.HTTP_201_CREATED)
async def push_zephyr_testcase(data: PushTestcaseRequest):
        """Push a test case to Zephyr, link to Jira, and add test steps."""
        try:
            project_key = data.jira_id.split("-")[0]
            testcase_name = data.testcase_name or f"Test Case for {data.jira_id}"
            objective = data.objective or f"To validate behavior for {data.jira_id}"

            async with httpx.AsyncClient() as client:
                # 1) Create test case
                created = await ZephyrService.create_testcase(project_key, testcase_name, objective, data.precondition, client)
                test_case_key = created.get("key")
                test_case_id = created.get("id")
                
                # Validate that test case was created successfully
                if not test_case_key or not test_case_id:
                    logger.error("Failed to create test case in Zephyr", created=created)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create test case in Zephyr Scale"
                    )

                # 2) Link to Jira issue (if Jira credentials available)
                linked = False
                issue_id = await JiraService.get_issue_id(data.jira_id, client)
                if issue_id:
                    linked = await ZephyrService.link_to_jira_issue(test_case_key, issue_id, client)

                # 3) Push steps
                steps_pushed = await ZephyrService.add_test_steps(test_case_key, data.test_steps, client)

            logger.info("Successfully pushed test case to Zephyr", 
                       test_case_key=test_case_key, 
                       jira_id=data.jira_id,
                       linked=linked, 
                       steps_pushed=steps_pushed)

            return PushTestcaseResponse(
                message="Test case created and steps pushed." if steps_pushed else "Test case created, but pushing steps failed.",
                testcase_key=test_case_key,
                testcase_id=test_case_id,
                linked_to_issue=linked,
                steps_pushed=steps_pushed
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to push test case to Zephyr", 
                        jira_id=data.jira_id, 
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to push test case to Zephyr Scale: {str(e)}"
            )
