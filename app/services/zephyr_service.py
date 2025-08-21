import structlog
import httpx
from base64 import b64encode
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.zephyr import PushTestcaseRequest, PushTestcaseResponse, TestStep
from app.services.test_case_service import TestCaseService
from app.core.dependencies import get_test_case_service
from app.config.settings import settings

logger = structlog.get_logger()

ZEPHYR_BASE_URL = settings.zephyr_base_url
ZEPHYR_API_TOKEN = settings.zephyr_api_token
JIRA_BASE_URL = settings.jira_base_url
JIRA_EMAIL = settings.jira_username
JIRA_API_TOKEN = settings.jira_api_token

router = APIRouter(prefix="/integrations", tags=["integrations"])

# GET endpoint to fetch JIRA ticket details by ticket key
@router.get("/jira/{ticket_key}")
async def get_jira_ticket(
    ticket_key: str,
    include_similar: bool = True,
    limit: int = 5,
    threshold: float = 0.7,
    tags: list[str] | None = None,
    priority: str | None = None,
    service: TestCaseService = Depends(get_test_case_service)
):
    """Fetch JIRA ticket details by ticket key (e.g., PROJ-123)"""
    try:
        logger.info("Fetching JIRA ticket", ticket_key=ticket_key)
        # Replace with actual service call to fetch ticket details
        ticket_data = await service.get_jira_ticket(
            ticket_key,
            include_similar=include_similar,
            limit=limit,
            threshold=threshold,
            tags=tags,
            priority=priority,
        )
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


@router.post("/zephyr/push", response_model=PushTestcaseResponse, status_code=status.HTTP_201_CREATED)
async def push_zephyr_testcase(data: PushTestcaseRequest):
    """Push a test case to Zephyr, link to Jira, and add test steps."""
    try:
        project_key = data.jira_id.split("-")[0]
        testcase_name = data.testcase_name or f"Test Case for {data.jira_id}"
        objective = data.objective or f"To validate behavior for {data.jira_id}"

        async with httpx.AsyncClient() as client:
            # 1) First get JIRA issue ID
            issue_id = await JiraService.get_issue_id(data.jira_id, client)
            if not issue_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"JIRA issue {data.jira_id} not found"
                )

            # 2) Check for duplicate test case
            is_duplicate = await ZephyrService.check_duplicate_testcase(
                issue_id=issue_id,
                testcase_name=testcase_name,
                client=client
            )

            if is_duplicate:
                # Generate suggested names
                suggested_names = [
                    f"{testcase_name} - v2",
                    f"{testcase_name} - Updated",
                    f"{testcase_name} - Revised",
                    f"{testcase_name} ({data.jira_id})",
                    f"{testcase_name} - Additional"
                ]
                
                logger.warning(
                    "Duplicate test case name detected",
                    jira_id=data.jira_id,
                    test_case_name=testcase_name
                )
                
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error": "Duplicate Test Case Name",
                        "message": f"A test case with the name '{testcase_name}' already exists for JIRA issue {data.jira_id}. Please edit the name and try again.",
                        "type": "DUPLICATE_TEST_CASE",
                        "user_action": "EDIT_NAME",
                        "original_name": testcase_name,
                        "jira_id": data.jira_id,
                        "suggested_names": suggested_names,
                        "instructions": "Choose one of the suggested names or create your own unique name for this test case."
                    }
                )

            # 3) Create test case if no duplicate found
            created = await ZephyrService.create_testcase(
                project_key, 
                testcase_name, 
                objective, 
                data.precondition, 
                client
            )
            test_case_key = created.get("key")
            test_case_id = created.get("id")
            
            if not test_case_key or not test_case_id:
                logger.error("Failed to create test case in Zephyr", created=created)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create test case in Zephyr Scale"
                )

            # 4) Link to Jira issue
            linked = await ZephyrService.link_to_jira_issue(test_case_key, issue_id, client)

            # 5) Push steps
            steps_pushed = await ZephyrService.add_test_steps(test_case_key, data.test_steps, client)

            logger.info(
                "Successfully pushed test case to Zephyr", 
                test_case_key=test_case_key, 
                jira_id=data.jira_id,
                linked=linked, 
                steps_pushed=steps_pushed
            )

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
        logger.error(
            "Failed to push test case to Zephyr", 
            jira_id=data.jira_id, 
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to push test case to Zephyr Scale: {str(e)}"
        )

class ZephyrService:
    @staticmethod
    def _headers() -> dict:
        """Generate headers for Zephyr API calls"""
        if not ZEPHYR_API_TOKEN:
            raise HTTPException(status_code=500, detail="Missing ZEPHYR_API_TOKEN")
        return {
            "Authorization": f"Bearer {ZEPHYR_API_TOKEN}",
            "Content-Type": "application/json"
        }

    @classmethod
    async def create_testcase(
        cls, 
        project_key: str, 
        name: str, 
        objective: str, 
        precondition: Optional[str], 
        client: httpx.AsyncClient
    ) -> Dict[str, Any]:
        """Create a new test case in Zephyr"""
        try:
            payload = {
                "projectKey": project_key,
                "name": name,
                "objective": objective,
                "status": "Draft"
            }
            if precondition:
                payload["precondition"] = precondition

            r = await client.post(
                f"{ZEPHYR_BASE_URL}/testcases", 
                headers=cls._headers(), 
                json=payload, 
                timeout=30
            )

            if r.status_code != 201:
                logger.error(
                    "Failed to create test case",
                    status_code=r.status_code,
                    response=r.text
                )
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Create test case failed: {r.text}"
                )

            return r.json()

        except Exception as e:
            logger.error("Error creating test case", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create test case: {str(e)}"
            )

    @classmethod
    async def get_testcases_for_issue(
        cls, 
        issue_id: int, 
        client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """Get all test cases linked to a JIRA issue"""
        try:
            url = f"{ZEPHYR_BASE_URL}/testcases"
            params = {
                "jql": f"issueId = {issue_id}",
                "maxResults": 100
            }

            r = await client.get(
                url,
                headers=cls._headers(),
                params=params,
                timeout=30
            )

            if r.status_code != 200:
                logger.error(
                    "Failed to fetch test cases", 
                    issue_id=issue_id,
                    status_code=r.status_code,
                    response=r.text
                )
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Failed to fetch test cases: {r.text}"
                )

            response_data = r.json()
            return response_data.get("values", [])

        except Exception as e:
            logger.error(
                "Error fetching test cases", 
                issue_id=issue_id,
                error=str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch test cases: {str(e)}"
            )

    @classmethod
    async def check_duplicate_testcase(
        cls, 
        issue_id: int, 
        testcase_name: str, 
        client: httpx.AsyncClient
    ) -> bool:
        """Check if a test case with same name exists for the issue"""
        try:
            existing_cases = await cls.get_testcases_for_issue(issue_id, client)
            normalized_name = testcase_name.strip().lower()

            for case in existing_cases:
                if case.get("name", "").strip().lower() == normalized_name:
                    logger.info(
                        "Duplicate test case found",
                        issue_id=issue_id,
                        test_case_name=testcase_name,
                        existing_case_key=case.get("key")
                    )
                    return True

            return False

        except Exception as e:
            logger.error(
                "Error checking for duplicates",
                issue_id=issue_id,
                test_case_name=testcase_name,
                error=str(e)
            )
            raise

    @classmethod
    async def link_to_jira_issue(
        cls, 
        test_case_key: str, 
        issue_id: int, 
        client: httpx.AsyncClient
    ) -> bool:
        """Link a test case to a JIRA issue"""
        try:
            link_url = f"{ZEPHYR_BASE_URL}/testcases/{test_case_key}/links/issues"
            body = {"issueId": issue_id}

            r = await client.post(
                link_url, 
                headers=cls._headers(), 
                json=body, 
                timeout=30
            )

            if r.status_code not in (200, 201):
                logger.error(
                    "Failed to link test case",
                    test_case_key=test_case_key,
                    issue_id=issue_id,
                    status_code=r.status_code,
                    response=r.text
                )
                return False

            return True

        except Exception as e:
            logger.error(
                "Error linking test case",
                test_case_key=test_case_key,
                issue_id=issue_id,
                error=str(e)
            )
            return False

    @classmethod
    async def add_test_steps(
        cls, 
        test_case_key: str, 
        steps: List[TestStep], 
        client: httpx.AsyncClient
    ) -> bool:
        """Add test steps to a test case"""
        try:
            items = [
                {
                    "inline": {
                        "description": s.step,
                        "testData": s.test_data,
                        "expectedResult": s.expected_result
                    }
                }
                for s in steps
            ]

            payload = {"mode": "OVERWRITE", "items": items}
            
            r = await client.post(
                f"{ZEPHYR_BASE_URL}/testcases/{test_case_key}/teststeps",
                headers=cls._headers(),
                json=payload,
                timeout=30
            )

            if r.status_code not in (200, 201):
                logger.error(
                    "Failed to add test steps",
                    test_case_key=test_case_key,
                    status_code=r.status_code,
                    response=r.text
                )
                return False

            return True

        except Exception as e:
            logger.error(
                "Error adding test steps",
                test_case_key=test_case_key,
                error=str(e)
            )
            return False

class JiraService:
    @staticmethod
    def _headers() -> dict:
        """Generate headers for JIRA API calls"""
        if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
            raise HTTPException(status_code=500, detail="Missing JIRA credentials")
        token = b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
        return {
            "Authorization": f"Basic {token}",
            "Accept": "application/json"
        }

    @classmethod
    async def get_issue_id(cls, issue_key: str, client: httpx.AsyncClient) -> Optional[int]:
        """Get JIRA issue ID from issue key"""
        try:
            if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
                return None

            url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=id"
            r = await client.get(url, headers=cls._headers(), timeout=30)

            if r.status_code == 200:
                return int(r.json().get("id"))

            logger.error(
                "Failed to get JIRA issue ID",
                issue_key=issue_key,
                status_code=r.status_code
            )
            return None

        except Exception as e:
            logger.error(
                "Error getting JIRA issue ID",
                issue_key=issue_key,
                error=str(e)
            )
            return None
