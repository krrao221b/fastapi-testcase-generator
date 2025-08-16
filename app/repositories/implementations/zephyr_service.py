import httpx
from typing import Optional, List, Dict, Any
import structlog
from app.repositories.interfaces.zephyr_service import IZephyrService
from app.models.schemas import TestCase
from app.config.settings import settings

logger = structlog.get_logger()


class ZephyrScaleService(IZephyrService):
    """Zephyr Scale (formerly Zephyr for JIRA) implementation"""
    
    def __init__(self):
        self.base_url = "https://eu.api.zephyrscale.smartbear.com/v2"
        self.api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjb250ZXh0Ijp7ImJhc2VVcmwiOiJodHRwczovL3Rlc3RjYXNlcmVhZHkuYXRsYXNzaWFuLm5ldCIsInVzZXIiOnsiYWNjb3VudElkIjoiNzEyMDIwOmFjMmQyZTA1LTBhOTYtNGNmYi1hMzQzLWZiZjM3MzBkMTVhYyIsInRva2VuSWQiOiJhN2QzNDdjNS0wZTgzLTQyMzgtOTAwNC1iYzk1OWE2OGMzYjMifX0sImlzcyI6ImNvbS5rYW5vYWgudGVzdC1tYW5hZ2VyIiwic3ViIjoiMTMxYmUzMjMtMGZmZC0zMzk1LTk5MGUtNjU0NDkxMTNhMTgzIiwiZXhwIjoxNzg2NjUwMzk0LCJpYXQiOjE3NTUxMTQzOTR9.-D_zJgOMtdspe5x7gL2j8Xsk-STZNVoeFNtFS3dRGUE"
    
    async def create_test_case(self, test_case: TestCase, project_key: str) -> Optional[str]:
        """Create a test case in Zephyr and return the test ID"""
        if not self._is_configured():
            logger.warning("Zephyr service not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "projectKey": project_key,
                    "name": test_case.title,
                    "objective": test_case.description,
                    "precondition": test_case.preconditions or "",
                    "priority": test_case.priority.value.upper(),
                    "status": "Draft",
                    "testScript": {
                        "type": "STEP_BY_STEP",
                        "steps": [
                            {
                                "description": step.action,
                                "testData": step.test_data or "",
                                "expectedResult": step.expected_result
                            }
                            for step in test_case.test_steps
                        ]
                    },
                    "labels": test_case.tags
                }
                
                headers = self._get_auth_headers()
                headers.update({
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                })
                
                response = await client.post(
                    f"{self.base_url}/rest/atm/1.0/testcase",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 201:
                    test_data = response.json()
                    test_id = test_data.get("key")
                    logger.info("Zephyr test case created successfully", test_id=test_id)
                    return test_id
                else:
                    logger.error("Failed to create Zephyr test case", 
                               status_code=response.status_code,
                               response=response.text)
                    return None
                    
        except Exception as e:
            logger.error("Error creating Zephyr test case", error=str(e))
            return None
    
    async def update_test_case(self, zephyr_test_id: str, test_case: TestCase) -> bool:
        """Update an existing test case in Zephyr"""
        if not self._is_configured():
            logger.warning("Zephyr service not configured")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "name": test_case.title,
                    "objective": test_case.description,
                    "precondition": test_case.preconditions or "",
                    "priority": test_case.priority.value.upper(),
                    "testScript": {
                        "type": "STEP_BY_STEP",
                        "steps": [
                            {
                                "description": step.action,
                                "testData": step.test_data or "",
                                "expectedResult": step.expected_result
                            }
                            for step in test_case.test_steps
                        ]
                    },
                    "labels": test_case.tags
                }
                
                headers = self._get_auth_headers()
                headers.update({
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                })
                
                response = await client.put(
                    f"{self.base_url}/rest/atm/1.0/testcase/{zephyr_test_id}",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info("Zephyr test case updated successfully", test_id=zephyr_test_id)
                    return True
                else:
                    logger.error("Failed to update Zephyr test case", 
                               test_id=zephyr_test_id,
                               status_code=response.status_code)
                    return False
                    
        except Exception as e:
            logger.error("Error updating Zephyr test case", test_id=zephyr_test_id, error=str(e))
            return False
    
    async def execute_test(self, zephyr_test_id: str, execution_status: str, comment: Optional[str] = None) -> bool:
        """Execute a test case in Zephyr with given status"""
        if not self._is_configured():
            logger.warning("Zephyr service not configured")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "testCaseKey": zephyr_test_id,
                    "status": execution_status.upper(),
                    "comment": comment or "",
                    "executedOn": None  # Current datetime will be used
                }
                
                headers = self._get_auth_headers()
                headers.update({
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                })
                
                response = await client.post(
                    f"{self.base_url}/rest/atm/1.0/testresult",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 201:
                    logger.info("Test execution recorded in Zephyr", 
                              test_id=zephyr_test_id, status=execution_status)
                    return True
                else:
                    logger.error("Failed to record test execution in Zephyr", 
                               test_id=zephyr_test_id,
                               status_code=response.status_code)
                    return False
                    
        except Exception as e:
            logger.error("Error executing test in Zephyr", 
                        test_id=zephyr_test_id, error=str(e))
            return False
    
    async def get_test_execution_history(self, zephyr_test_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a test case"""
        if not self._is_configured():
            logger.warning("Zephyr service not configured")
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                headers = self._get_auth_headers()
                headers["Accept"] = "application/json"
                
                response = await client.get(
                    f"{self.base_url}/rest/atm/1.0/testcase/{zephyr_test_id}/testresults",
                    headers=headers
                )
                
                if response.status_code == 200:
                    results = response.json()
                    logger.info("Test execution history retrieved", 
                              test_id=zephyr_test_id, count=len(results))
                    return results
                else:
                    logger.error("Failed to get test execution history", 
                               test_id=zephyr_test_id,
                               status_code=response.status_code)
                    return []
                    
        except Exception as e:
            logger.error("Error getting test execution history", 
                        test_id=zephyr_test_id, error=str(e))
            return []
    
    async def create_test_cycle(self, name: str, project_key: str, version_id: str) -> Optional[str]:
        """Create a test cycle in Zephyr"""
        if not self._is_configured():
            logger.warning("Zephyr service not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "name": name,
                    "projectKey": project_key,
                    "versionId": version_id,
                    "description": f"Test cycle created for {name}"
                }
                
                headers = self._get_auth_headers()
                headers.update({
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                })
                
                response = await client.post(
                    f"{self.base_url}/rest/atm/1.0/testcycle",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 201:
                    cycle_data = response.json()
                    cycle_id = cycle_data.get("key")
                    logger.info("Test cycle created in Zephyr", cycle_id=cycle_id)
                    return cycle_id
                else:
                    logger.error("Failed to create test cycle in Zephyr", 
                               status_code=response.status_code)
                    return None
                    
        except Exception as e:
            logger.error("Error creating test cycle in Zephyr", error=str(e))
            return None
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Zephyr API"""
        return {
            "Authorization": f"Bearer {self.api_token}"
        }
    
    def _is_configured(self) -> bool:
        """Check if Zephyr service is properly configured"""
        return bool(self.base_url and self.api_token)
