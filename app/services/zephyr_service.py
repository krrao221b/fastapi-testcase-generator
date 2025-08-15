from base64 import b64encode
from typing import Optional, List
import httpx
from fastapi import HTTPException
from app.models.zephyr import TestStep
from app.config.settings import settings

ZEPHYR_BASE_URL = settings.zephyr_base_url
ZEPHYR_API_TOKEN = settings.zephyr_api_token
JIRA_BASE_URL = settings.jira_base_url
JIRA_EMAIL = settings.jira_username
JIRA_API_TOKEN = settings.jira_api_token

class ZephyrService:
    @staticmethod
    def _headers() -> dict:
        if not ZEPHYR_API_TOKEN:
            raise HTTPException(status_code=500, detail="Missing ZEPHYR_API_TOKEN")
        return {
            "Authorization": f"Bearer {ZEPHYR_API_TOKEN}",
            "Content-Type": "application/json"
        }

    @classmethod
    async def create_testcase(cls, project_key: str, name: str, objective: str, precondition: Optional[str], client: httpx.AsyncClient) -> dict:
        payload = {
            "projectKey": project_key,
            "name": name,
            "objective": objective
        }
        if precondition:
            payload["precondition"] = precondition
        r = await client.post(f"{ZEPHYR_BASE_URL}/testcases", headers=cls._headers(), json=payload, timeout=30)
        if r.status_code != 201:
            raise HTTPException(status_code=r.status_code, detail=f"Create test case failed: {r.text}")
        return r.json()

    @classmethod
    async def link_to_jira_issue(cls, test_case_key: str, issue_id: int, client: httpx.AsyncClient) -> bool:
        link_url = f"{ZEPHYR_BASE_URL}/testcases/{test_case_key}/links/issues"
        body = {"issueId": issue_id}
        r = await client.post(link_url, headers=cls._headers(), json=body, timeout=30)
        return r.status_code in (200, 201)

    @classmethod
    async def add_test_steps(cls, test_case_key: str, steps: List[TestStep], client: httpx.AsyncClient) -> bool:
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
        r = await client.post(f"{ZEPHYR_BASE_URL}/testcases/{test_case_key}/teststeps", headers=cls._headers(), json=payload, timeout=30)
        return r.status_code in (200, 201)

class JiraService:
    @staticmethod
    def _headers() -> dict:
        if not (JIRA_BASE_URL and JIRA_EMAIL and JIRA_API_TOKEN):
            raise HTTPException(status_code=500, detail="Missing Jira credentials")
        token = b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    @classmethod
    async def get_issue_id(cls, issue_key: str, client: httpx.AsyncClient) -> Optional[int]:
        if not (JIRA_BASE_URL and JIRA_EMAIL and JIRA_API_TOKEN):
            return None
        url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}?fields=id"
        r = await client.get(url, headers=cls._headers(), timeout=30)
        if r.status_code == 200:
            return int(r.json().get("id"))
        return None
