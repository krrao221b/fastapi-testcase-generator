import google.generativeai as genai
from typing import List
from datetime import datetime
import json
import structlog
from app.repositories.interfaces.ai_service import IAIService
from app.models.schemas import GenerateTestCaseRequest, TestCase, TestStep, TestCaseStatus
from app.config.settings import settings

logger = structlog.get_logger()


class GeminiService(IAIService):
    """Google Gemini implementation of AI service"""
    
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.embedding_model = 'models/text-embedding-004'
    
    async def generate_test_case(self, request: GenerateTestCaseRequest) -> TestCase:
        """Generate a test case using Google Gemini"""
        logger.info("Starting Gemini test case generation", 
                   feature=request.feature_description[:100])
        
        try:
            # Check if API key is configured
            if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
                logger.error("Gemini API key not configured properly")
                return self._create_fallback_test_case(request)
            
            prompt = self._build_optimized_prompt(request)
            logger.info("Built prompt for Gemini", prompt_length=len(prompt))
            
            # Generate content using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                )
            )
            
            # Check if response has valid content
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("Gemini returned empty response", 
                              finish_reason=response.candidates[0].finish_reason if response.candidates else "No candidates")
                return self._create_fallback_test_case(request)
            
            generated_content = response.text
            logger.info("Received Gemini response", 
                       content_length=len(generated_content),
                       model_used=settings.gemini_model)
            
            test_case_data = self._parse_generated_test_case(generated_content, request)
            
            logger.info("Test case generated successfully using Gemini", 
                       feature_description=request.feature_description)
            
            return TestCase(**test_case_data)
            
        except Exception as e:
            logger.error("Failed to generate test case using Gemini", 
                        feature_description=request.feature_description, 
                        error=str(e),
                        error_type=type(e).__name__)
            # Return a basic test case as fallback
            return self._create_fallback_test_case(request)
            
            generated_content = response.text
            logger.info("Received Gemini response", 
                       content_length=len(generated_content),
                       model_used="gemini-1.5-flash")
            
            test_case_data = self._parse_generated_test_case(generated_content, request)
            
            logger.info("Test case generated successfully using Gemini", 
                       feature_description=request.feature_description)
            
            return TestCase(**test_case_data)
            
        except Exception as e:
            logger.error("Failed to generate test case using Gemini", 
                        feature_description=request.feature_description, 
                        error=str(e),
                        error_type=type(e).__name__)
            # Return a basic test case as fallback
            return self._create_fallback_test_case(request)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using Gemini"""
        try:
            # Truncate text to save tokens
            truncated_text = text[:500]
            
            response = genai.embed_content(
                model=self.embedding_model,
                content=truncated_text
            )
            return response['embedding']
            
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            return []
    
    async def improve_test_case(self, test_case: TestCase, feedback: str) -> TestCase:
        """Improve an existing test case based on feedback using Gemini"""
        try:
            prompt = f"Improve this test case based on feedback: {feedback}\n\nTest Case: {test_case.title}\nSteps: {[step.action for step in test_case.test_steps]}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=250,
                    temperature=0.3,
                    top_p=0.9,
                )
            )
            
            if response.text:
                # For simplicity, just update the description with the improvement
                test_case.description = f"Improved: {response.text[:200]}"
                test_case.updated_at = datetime.utcnow()
            
            logger.info("Test case improved successfully using Gemini", test_case_id=test_case.id)
            return test_case
            
        except Exception as e:
            logger.error("Failed to improve test case using Gemini", 
                        test_case_id=test_case.id, error=str(e))
            return test_case  # Return original if improvement fails
    
    def _build_optimized_prompt(self, request: GenerateTestCaseRequest) -> str:
        """Build optimized prompt for better test case generation"""
        
        system_prompt = """Generate a structured test case in JSON format. Respond ONLY with valid JSON:

{
  "title": "Clear test case title",
  "description": "Brief test description",
  "test_steps": [
    {
      "step_number": 1,
      "action": "Specific action to perform",
      "expected_result": "Expected outcome"
    }
  ],
  "expected_result": "Overall expected result",
  "preconditions": "Required setup"
}"""

        user_prompt = f"""Feature: {request.feature_description}
Acceptance Criteria: {request.acceptance_criteria}
Priority: {request.priority.value}"""
        
        if request.additional_context:
            user_prompt += f"\nContext: {request.additional_context}"
        
        if request.tags:
            user_prompt += f"\nTags: {', '.join(request.tags)}"
            
        return f"{system_prompt}\n\n{user_prompt}"
        
        return prompt
    
    def _parse_generated_test_case(self, generated_content: str, request: GenerateTestCaseRequest) -> dict:
        """Parse generated content into test case data with JSON support"""
        try:
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', generated_content, re.DOTALL)
            
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                    
                    # Process test steps
                    test_steps = []
                    for i, step_data in enumerate(parsed_json.get("test_steps", []), 1):
                        if isinstance(step_data, dict):
                            test_steps.append(TestStep(
                                step_number=step_data.get("step_number", i),
                                action=step_data.get("action", ""),
                                expected_result=step_data.get("expected_result", "Action completed successfully")
                            ))
                    
                    # If no steps found, create a default one
                    if not test_steps:
                        test_steps = [TestStep(
                            step_number=1,
                            action=f"Test {request.feature_description}",
                            expected_result=request.acceptance_criteria
                        )]
                    
                    return {
                        "id": 0,
                        "title": parsed_json.get("title", f"Test Case: {request.feature_description[:50]}"),
                        "description": parsed_json.get("description", "AI-generated test case"),
                        "feature_description": request.feature_description,
                        "acceptance_criteria": request.acceptance_criteria,
                        "priority": request.priority,
                        "status": TestCaseStatus.DRAFT,
                        "tags": request.tags,
                        "preconditions": parsed_json.get("preconditions", "System ready for testing"),
                        "test_steps": test_steps,
                        "expected_result": parsed_json.get("expected_result", "Test should pass successfully"),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, falling back to text parsing")
            
            # Fallback: text-based parsing
            return self._parse_text_response(generated_content, request)
            
        except Exception as e:
            logger.error("Failed to parse generated test case", error=str(e))
            return self._create_fallback_test_case(request).__dict__

    def _parse_text_response(self, generated_content: str, request: GenerateTestCaseRequest) -> dict:
        """Fallback text parsing for non-JSON responses"""
        lines = generated_content.strip().split('\n')
        
        title = f"Test Case: {request.feature_description[:50]}"
        description = generated_content[:200] + "..." if len(generated_content) > 200 else generated_content
        expected_result = "Test should pass successfully"
        preconditions = "System ready for testing"
        test_steps = []
        
        # Create basic test steps from the content
        step_indicators = ["step", "action", "test", "verify", "check", "enter", "click", "submit"]
        step_num = 1
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in step_indicators) and len(line) > 10:
                test_steps.append(TestStep(
                    step_number=step_num,
                    action=line,
                    expected_result="Action completed successfully"
                ))
                step_num += 1
                if step_num > 5:  # Limit to 5 steps
                    break
        
        # If no steps found, create a basic one
        if not test_steps:
            test_steps = [
                TestStep(
                    step_number=1,
                    action=f"Test {request.feature_description}",
                    expected_result=request.acceptance_criteria
                )
            ]
        try:
            return {
                "id": 0,
                "title": title,
                "description": description,
                "feature_description": request.feature_description,
                "acceptance_criteria": request.acceptance_criteria,
                "priority": request.priority,
                "status": TestCaseStatus.DRAFT,
                "tags": request.tags,
                "preconditions": preconditions,
                "test_steps": test_steps,
                "expected_result": expected_result,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to parse generated test case", error=str(e))
            return self._create_fallback_test_case(request).__dict__
    
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
                    expected_result="Feature works according to acceptance criteria"
                )
            ],
            expected_result="Test passes successfully",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )