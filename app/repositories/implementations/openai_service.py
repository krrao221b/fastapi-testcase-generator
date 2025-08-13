import asyncio
import json
from typing import List
from datetime import datetime
from openai import OpenAI
import structlog
from app.repositories.interfaces.ai_service import IAIService
from app.models.schemas import GenerateTestCaseRequest, TestCase, TestStep, TestCaseStatus
from app.config.settings import settings

logger = structlog.get_logger()

class OpenAIService(IAIService):
    """GitHub Copilot Models API implementation of AI service"""
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key
        )
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model
        
    async def generate_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        """Generate a test case using Copilot Models API (async wrapper)"""
        def sync_call():
            try:
                prompt = self._build_test_case_prompt(request)
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    top_p=0.9,
                    model=self.model
                )
                generated_content = response.choices[0].message.content or ""
                parse_test_case_data = self._parse_generated_test_case(generated_content, request)
                return TestCase(**parsed)
            except Exception as e:
                logger.error("Failed to generate test case", error=str(e))
                return self._create_fallback_test_case(request)
        return await asyncio.get_event_loop().run_in_executor(None, sync_call)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using Copilot Models API (async wrapper)"""
        def sync_call():
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error("Failed to generate embedding", error=str(e))
                return []
        return await asyncio.get_event_loop().run_in_executor(None, sync_call)

    async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
        """Improve an existing test case based on feedback using Copilot Models API (async wrapper)"""
        def sync_call():
            try:
                prompt = self._build_improvement_prompt(test_case, feedback)
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self._get_improvement_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    top_p=0.9,
                    model=self.model
                )
                improved_content = response.choices[0].message.content or ""
                parsed = self._parse_improved_test_case(improved_content, test_case)
                return TestCase(**parsed)
            except Exception as e:
                logger.error("Failed to improve test case", error=str(e))
                return test_case
        return await asyncio.get_event_loop().run_in_executor(None, sync_call)
    
    # async def generate_embedding(self, text: str) -> List[float]:
    #     """Generate embedding for a given text"""
    #     try:
    #         response = await self.client.embeddings.create(
    #             model=self.embedding_model,
    #             input=text
    #         )
    #         return response.data[0].embedding
            
    #     except Exception as e:
    #         logger.error("Failed to generate embedding", text=text[:100], error=str(e))
    #         return []
    
    # async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
    #     """Improve an existing test case based on feedback"""
    #     try:
    #         prompt = self._build_improvement_prompt(test_case, feedback)
            
    #         response = await self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": self._get_improvement_system_prompt()},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             temperature=0.5,
    #             max_tokens=2000
    #         )
            
    #         improved_content = response.choices[0].message.content
    #         improved_data = self._parse_improved_test_case(improved_content, test_case)
            
    #         logger.info("Test case improved successfully", test_case_id=test_case.id)
            
    #         return TestCase(**improved_data)
            
    #     except Exception as e:
    #         logger.error("Failed to improve test case", 
    #                     test_case_id=test_case.id, error=str(e))
    #         return test_case  # Return original if improvement fails
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for test case generation"""
        return """You are an expert test case generator. Generate comprehensive, well-structured test cases based on feature descriptions and acceptance criteria.

Your response should be in JSON format with the following structure:
{
  "title": "Clear, concise test case title",
  "description": "Detailed description of what this test case validates",
  "test_steps": [
    {
      "step_number": 1,
      "action": "Action to perform",
      "expected_result": "Expected outcome",
      "test_data": "Any test data needed (optional)"
    }
  ],
  "expected_result": "Overall expected result of the test case",
  "preconditions": "Any setup or preconditions needed"
}

Ensure test cases are:
- Clear and unambiguous
- Include specific test steps
- Cover edge cases when relevant
- Follow testing best practices
- Include appropriate test data"""
    
    def _get_improvement_system_prompt(self) -> str:
        """Get system prompt for test case improvement"""
        return """You are an expert test case reviewer. Improve existing test cases based on feedback while maintaining their core purpose.

Focus on:
- Clarity and readability
- Coverage of edge cases
- Test step specificity
- Better test data
- Addressing the specific feedback provided

Return the improved test case in the same JSON format as the original."""
    
    def _build_test_case_prompt(self, request: GenerateTestCaseRequest) -> str:
        """Build prompt for test case generation"""
        prompt = f"""Generate a comprehensive test case for the following:

Feature Description:
{request.feature_description}

Acceptance Criteria:
{request.acceptance_criteria}

Priority: {request.priority.value}
"""
        
        if request.additional_context:
            prompt += f"\nAdditional Context:\n{request.additional_context}"
        
        if request.tags:
            prompt += f"\nTags: {', '.join(request.tags)}"
        
        return prompt
    
    def _build_improvement_prompt(self, test_case: TestCase, feedback: str) -> str:
        """Build prompt for test case improvement"""
        current_test_case = {
            "title": test_case.title,
            "description": test_case.description,
            "test_steps": [step.dict() for step in test_case.test_steps],
            "expected_result": test_case.expected_result,
            "preconditions": test_case.preconditions
        }
        
        return f"""Improve the following test case based on the feedback provided:

Current Test Case:
{json.dumps(current_test_case, indent=2)}

Feedback:
{feedback}

Please provide an improved version addressing the feedback while maintaining the test case's core objectives."""
    
    def _parse_generated_test_case(self, generated_content: str, request: GenerateTestCaseRequest) -> dict:
        """Parse generated content into test case data"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', generated_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                # Fallback to manual parsing if JSON extraction fails
                parsed_data = self._manual_parse_test_case(generated_content)
            
            # Convert test_steps to TestStep objects
            test_steps = []
            for step_data in parsed_data.get("test_steps", []):
                if 'test_data' not in step_data:
                    step_data['test_data'] = None
                if isinstance(step_data['test_data'], (dict, list)):
                    step_data['test_data'] = json.dumps(step_data['test_data'])
                test_steps.append(TestStep(**step_data))
            
            return {
                "id": 0,  # Will be set by database
                "title": parsed_data.get("title", "Generated Test Case"),
                "description": parsed_data.get("description", "AI-generated test case"),
                "feature_description": request.feature_description,
                "acceptance_criteria": request.acceptance_criteria,
                "priority": request.priority,
                "status": TestCaseStatus.DRAFT,
                "tags": request.tags,
                "preconditions": parsed_data.get("preconditions", ""),
                "test_steps": test_steps,
                "expected_result": parsed_data.get("expected_result", ""),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Failed to parse generated test case", error=str(e))
            return self._create_fallback_test_case(request).__dict__
    
    def _parse_improved_test_case(self, improved_content: str, original: TestCase) -> dict:
        """Parse improved content into test case data"""
        try:
            import re
            json_match = re.search(r'\{.*\}', improved_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                return original.__dict__
            
            # Convert test_steps to TestStep objects
            test_steps = []
            for step_data in parsed_data.get("test_steps", []):
                if 'test_data' not in step_data:
                    step_data['test_data'] = None
                if isinstance(step_data['test_data'], (dict, list)):
                    step_data['test_data'] = json.dumps(step_data['test_data'])
                test_steps.append(TestStep(**step_data))
            
            # Preserve original data and update with improvements
            improved_data = original.__dict__.copy()
            improved_data.update({
                "title": parsed_data.get("title", original.title),
                "description": parsed_data.get("description", original.description),
                "test_steps": test_steps,
                "expected_result": parsed_data.get("expected_result", original.expected_result),
                "preconditions": parsed_data.get("preconditions", original.preconditions),
                "updated_at": datetime.utcnow()
            })
            
            return improved_data
            
        except Exception as e:
            logger.error("Failed to parse improved test case", error=str(e))
            return original.__dict__
    
    def _manual_parse_test_case(self, content: str) -> dict:
        """Manual parsing fallback for non-JSON responses"""
        # This is a simplified implementation
        # In production, you might want more sophisticated parsing
        return {
            "title": "Generated Test Case",
            "description": content[:200] + "..." if len(content) > 200 else content,
            "test_steps": [
                {
                    "step_number": 1,
                    "action": "Execute the feature",
                    "expected_result": "Feature works as expected",
                    "test_data": None
                }
            ],
            "expected_result": "Test passes successfully",
            "preconditions": "System is ready for testing"
        }
    
    def _create_fallback_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        """Create a basic fallback test case"""
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
                    test_data=None
                )
            ],
            expected_result="Test passes successfully",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
