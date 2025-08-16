import json
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
import structlog

from app.config.settings import settings
from app.models.schemas import (
    GenerateTestCaseRequest,
    TestCase,
    TestCaseStatus,
    TestStep,
)
from app.repositories.interfaces.ai_service import IAIService

logger = structlog.get_logger()


class GeminiService(IAIService):
    """Google Gemini implementation of AI service (used as fallback)."""

    def __init__(self) -> None:
        self.model: Optional[genai.GenerativeModel] = None
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        self.embedding_model = "models/text-embedding-004"

    async def generate_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        try:
            if not self.model:
                logger.error("Gemini model not configured; using basic fallback")
                return self._create_fallback_test_case(request)

            prompt = self._build_generation_prompt(request)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            text = getattr(response, "text", None) or ""
            if not text:
                return self._create_fallback_test_case(request)
            data = self._parse_generated_test_case(text, request)
            return TestCase(**data)
        except Exception as e:
            logger.error("Gemini generate_test_case failed", error=str(e))
            return self._create_fallback_test_case(request)

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            if not settings.gemini_api_key:
                return []
            response = genai.embed_content(model=self.embedding_model, content=text[:500])
            return response.get("embedding", [])
        except Exception as e:
            logger.error("Gemini embedding failed", error=str(e))
            return []

    async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
        try:
            if not self.model:
                return test_case
            prompt = self._build_improvement_prompt(test_case, feedback)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            text = getattr(response, "text", None)
            if not text:
                return test_case
            parsed = self._parse_improved_test_case(text, test_case)
            return TestCase(**parsed)
        except Exception as e:
            logger.error("Gemini improve_test_case failed", error=str(e))
            return test_case

    def _build_generation_prompt(self, request: GenerateTestCaseRequest) -> str:
        system = (
            """Generate a structured test case in JSON format. Respond ONLY with valid JSON:
{
  \"title\": \"Clear test case title\",
  \"description\": \"Brief test description\",
  \"test_steps\": [
    {
      \"step_number\": 1,
      \"action\": \"Specific action to perform\",
      \"expected_result\": \"Expected outcome\",
      \"test_data\": null
    }
  ],
  \"expected_result\": \"Overall expected result\",
  \"preconditions\": \"Required setup\"
}
"""
        )
        user = (
            f"""Feature: {request.feature_description}
Acceptance Criteria: {request.acceptance_criteria}
Priority: {request.priority.value}"""
        )
        if request.additional_context:
            user += f"\nContext: {request.additional_context}"
        if request.tags:
            user += f"\nTags: {', '.join(request.tags)}"
        return f"{system}\n\n{user}"

    def _build_improvement_prompt(self, test_case: TestCase, feedback: str) -> str:
        current = {
            "title": test_case.title,
            "description": test_case.description,
            "test_steps": [step.dict() for step in test_case.test_steps],
            "expected_result": test_case.expected_result,
            "preconditions": test_case.preconditions,
        }
        return (
            "Improve the following test case based on the feedback provided. Return ONLY JSON in the same schema.\n\n"
            f"Current Test Case:\n{json.dumps(current, indent=2)}\n\nFeedback:\n{feedback}"
        )

    def _parse_generated_test_case(self, content: str, request: GenerateTestCaseRequest) -> dict:
        try:
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(m.group()) if m else self._manual_parse_test_case(content)

            steps: List[TestStep] = []
            for step in parsed.get("test_steps", []):
                if "test_data" not in step:
                    step["test_data"] = None
                if isinstance(step["test_data"], (dict, list)):
                    step["test_data"] = json.dumps(step["test_data"])  # normalize
                steps.append(TestStep(**step))

            return {
                "id": 0,
                "title": parsed.get("title", f"Test Case: {request.feature_description[:50]}"),
                "description": parsed.get("description", "AI-generated test case"),
                "feature_description": request.feature_description,
                "acceptance_criteria": request.acceptance_criteria,
                "priority": request.priority,
                "status": TestCaseStatus.DRAFT,
                "tags": request.tags,
                "preconditions": parsed.get("preconditions", "System ready for testing"),
                "test_steps": steps or [
                    TestStep(step_number=1, action=f"Test {request.feature_description}", expected_result=request.acceptance_criteria, test_data=None)
                ],
                "expected_result": parsed.get("expected_result", "Test should pass successfully"),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "jira_issue_key": request.jira_issue_key,
            }
        except Exception as e:
            logger.error("Gemini parse generated failed", error=str(e))
            return self._create_fallback_test_case(request).__dict__

    def _parse_improved_test_case(self, content: str, original: TestCase) -> dict:
        try:
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return original.__dict__
            parsed = json.loads(m.group())

            steps: List[TestStep] = []
            for step in parsed.get("test_steps", []):
                if "test_data" not in step:
                    step["test_data"] = None
                if isinstance(step["test_data"], (dict, list)):
                    step["test_data"] = json.dumps(step["test_data"])  # normalize
                steps.append(TestStep(**step))

            improved = original.__dict__.copy()
            improved.update(
                {
                    "title": parsed.get("title", original.title),
                    "description": parsed.get("description", original.description),
                    "test_steps": steps or original.test_steps,
                    "expected_result": parsed.get("expected_result", original.expected_result),
                    "preconditions": parsed.get("preconditions", original.preconditions),
                    "updated_at": datetime.utcnow(),
                }
            )
            return improved
        except Exception as e:
            logger.error("Gemini parse improved failed", error=str(e))
            return original.__dict__

    def _manual_parse_test_case(self, content: str) -> dict:
        return {
            "title": "Generated Test Case",
            "description": (content[:200] + "...") if len(content) > 200 else content,
            "test_steps": [
                {
                    "step_number": 1,
                    "action": "Execute the feature",
                    "expected_result": "Feature works as expected",
                    "test_data": None,
                }
            ],
            "expected_result": "Test passes successfully",
            "preconditions": "System is ready for testing",
        }

    def _create_fallback_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        return TestCase(
            id=0,
            title=f"Test Case for {request.feature_description[:50]}...",
            description="AI-generated test case (fallback)",
            feature_description=request.feature_description,
            acceptance_criteria=request.acceptance_criteria,
            priority=request.priority,
            status=TestCaseStatus.DRAFT,
            tags=request.tags,
            preconditions="System is ready for testing",
            test_steps=[
                TestStep(
                    step_number=1,
                    action="Execute the feature as described",
                    expected_result="Feature works according to acceptance criteria",
                    test_data=None,
                )
            ],
            expected_result="Test passes successfully",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            jira_issue_key=request.jira_issue_key,
        )
import json
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
import structlog

from app.config.settings import settings
from app.models.schemas import (
    GenerateTestCaseRequest,
    TestCase,
    TestCaseStatus,
    TestStep,
)
from app.repositories.interfaces.ai_service import IAIService

logger = structlog.get_logger()


class GeminiService(IAIService):
    """Google Gemini implementation of AI service (used as fallback)."""

    def __init__(self) -> None:
        self.model: Optional[genai.GenerativeModel] = None
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        self.embedding_model = "models/text-embedding-004"

    async def generate_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        logger.info(
            "Starting Gemini test case generation",
            feature=request.feature_description[:100],
        )
        try:
            if not self.model:
                logger.error("Gemini model not configured; using fallback")
                return self._create_fallback_test_case(request)

            prompt = self._build_optimized_prompt(request)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                ),
            )

            text = getattr(response, "text", None) or ""
            if not text:
                logger.warning("Gemini returned empty text; using fallback")
                return self._create_fallback_test_case(request)

            data = self._parse_generated_test_case(text, request)
            return TestCase(**data)
        except Exception as e:
            logger.error(
                "Failed to generate test case using Gemini",
                error=str(e),
                error_type=type(e).__name__,
            )
            return self._create_fallback_test_case(request)

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            if not settings.gemini_api_key:
                logger.error("Gemini API key not configured; cannot embed")
                return []
            truncated = text[:500]
            response = genai.embed_content(model=self.embedding_model, content=truncated)
            return response["embedding"]
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            return []

    async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
        try:
            if not self.model:
                logger.error("Gemini model not configured; returning original test case")
                return test_case
            prompt = self._build_improvement_prompt(test_case, feedback)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            text = getattr(response, "text", None)
            if text:
                parsed = self._parse_improved_test_case(text, test_case)
                return TestCase(**parsed)
            return test_case
        except Exception as e:
            logger.error("Failed to improve test case using Gemini", error=str(e))
            return test_case

    def _build_optimized_prompt(self, request: GenerateTestCaseRequest) -> str:
        system_prompt = (
            """Generate a structured test case in JSON format. Respond ONLY with valid JSON:
{
  \"title\": \"Clear test case title\",
  \"description\": \"Brief test description\",
  \"test_steps\": [
    {
      \"step_number\": 1,
      \"action\": \"Specific action to perform\",
      \"expected_result\": \"Expected outcome\",
      \"test_data\": null
    }
  ],
  \"expected_result\": \"Overall expected result\",
  \"preconditions\": \"Required setup\"
}
"""
        )
        user_prompt = (
            f"""Feature: {request.feature_description}
Acceptance Criteria: {request.acceptance_criteria}
Priority: {request.priority.value}"""
        )
        if request.additional_context:
            user_prompt += f"\nContext: {request.additional_context}"
        if request.tags:
            user_prompt += f"\nTags: {', '.join(request.tags)}"
        return f"{system_prompt}\n\n{user_prompt}"

    def _build_improvement_prompt(self, test_case: TestCase, feedback: str) -> str:
        current = {
            "title": test_case.title,
            "description": test_case.description,
            "test_steps": [step.dict() for step in test_case.test_steps],
            "expected_result": test_case.expected_result,
            "preconditions": test_case.preconditions,
        }
        return (
            "Improve the following test case based on the feedback provided. Return ONLY JSON in the same schema.\n\n"
            f"Current Test Case:\n{json.dumps(current, indent=2)}\n\nFeedback:\n{feedback}"
        )

    def _parse_generated_test_case(self, content: str, request: GenerateTestCaseRequest) -> dict:
        try:
            import re

            m = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(m.group()) if m else self._manual_parse_test_case(content)

            steps: List[TestStep] = []
            for i, step in enumerate(parsed.get("test_steps", []), start=1):
                if "test_data" not in step:
                    step["test_data"] = None
                if isinstance(step["test_data"], (dict, list)):
                    step["test_data"] = json.dumps(step["test_data"])  # normalize
                steps.append(TestStep(**step))

            return {
                "id": 0,
                "title": parsed.get("title", f"Test Case: {request.feature_description[:50]}"),
                "description": parsed.get("description", "AI-generated test case"),
                "feature_description": request.feature_description,
                "acceptance_criteria": request.acceptance_criteria,
                "priority": request.priority,
                "status": TestCaseStatus.DRAFT,
                "tags": request.tags,
                "preconditions": parsed.get("preconditions", "System ready for testing"),
                "test_steps": steps or [
                    TestStep(step_number=1, action=f"Test {request.feature_description}", expected_result=request.acceptance_criteria, test_data=None)
                ],
                "expected_result": parsed.get("expected_result", "Test should pass successfully"),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "jira_issue_key": request.jira_issue_key,
            }
        except Exception as e:
            logger.error("Failed to parse generated test case (Gemini)", error=str(e))
            return self._create_fallback_test_case(request).__dict__

    def _parse_improved_test_case(self, content: str, original: TestCase) -> dict:
        try:
            import re

            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return original.__dict__
            parsed = json.loads(m.group())

            steps: List[TestStep] = []
            for step in parsed.get("test_steps", []):
                if "test_data" not in step:
                    step["test_data"] = None
                if isinstance(step["test_data"], (dict, list)):
                    step["test_data"] = json.dumps(step["test_data"])  # normalize
                steps.append(TestStep(**step))

            improved = original.__dict__.copy()
            improved.update(
                {
                    "title": parsed.get("title", original.title),
                    "description": parsed.get("description", original.description),
                    "test_steps": steps or original.test_steps,
                    "expected_result": parsed.get("expected_result", original.expected_result),
                    "preconditions": parsed.get("preconditions", original.preconditions),
                    "updated_at": datetime.utcnow(),
                }
            )
            return improved
        except Exception as e:
            logger.error("Failed to parse improved test case (Gemini)", error=str(e))
            return original.__dict__

    def _manual_parse_test_case(self, content: str) -> dict:
        return {
            "title": "Generated Test Case",
            "description": (content[:200] + "...") if len(content) > 200 else content,
            "test_steps": [
                {
                    "step_number": 1,
                    "action": "Execute the feature",
                    "expected_result": "Feature works as expected",
                    "test_data": None,
                }
            ],
            "expected_result": "Test passes successfully",
            "preconditions": "System is ready for testing",
        }

    def _create_fallback_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        return TestCase(
            id=0,
            title=f"Test Case for {request.feature_description[:50]}...",
            description="AI-generated test case (fallback)",
            feature_description=request.feature_description,
            acceptance_criteria=request.acceptance_criteria,
            priority=request.priority,
            status=TestCaseStatus.DRAFT,
            tags=request.tags,
            preconditions="System is ready for testing",
            test_steps=[
                TestStep(
                    step_number=1,
                    action="Execute the feature as described",
                    expected_result="Feature works according to acceptance criteria",
                    test_data=None,
                )
            ],
            expected_result="Test passes successfully",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            jira_issue_key=request.jira_issue_key,
        )