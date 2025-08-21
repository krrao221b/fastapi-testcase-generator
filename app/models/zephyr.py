from typing import Optional, List
from pydantic import BaseModel

class TestStep(BaseModel):
    step: str
    test_data: str
    expected_result: str

class PushTestcaseRequest(BaseModel):
    jira_id: str
    testcase_name: Optional[str] = None
    objective: Optional[str] = None
    precondition: Optional[str] = None
    test_steps: List[TestStep]

class PushTestcaseResponse(BaseModel):
    message: str
    testcase_key: str
    testcase_id: int
    linked_to_issue: bool
    steps_pushed: bool
