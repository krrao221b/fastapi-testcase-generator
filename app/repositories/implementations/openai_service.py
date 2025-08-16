import asyncio
import json
from typing import List
from datetime import datetime
from openai import OpenAI
import structlog
import traceback
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
                # Visible debug: show client config (mask API key)
                try:
                    api_key = getattr(self.client, 'api_key', None)
                    if api_key:
                        masked = api_key[:4] + '...' + api_key[-4:]
                    else:
                        masked = None
                except Exception:
                    masked = None
                print("[debug] OpenAI client base:", getattr(self.client, 'base_url', settings.openai_base_url))
                print("[debug] OpenAI client api_key (masked):", masked)
                logger.info("OpenAI client info", base=getattr(self.client, 'base_url', settings.openai_base_url), api_key_masked=masked, model=self.model)

                prompt = self._build_test_case_prompt(request)
                # Print a short preview of prompt to stdout (first 400 chars)
                print("[debug] Prompt preview:", prompt[:400].replace('\n', ' '))
                logger.info("Prompt built", prompt_preview=prompt[:200])

                print("[debug] Calling OpenAI chat.completions.create()")
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    top_p=0.9,
                    model=self.model
                )

                print("[debug] OpenAI call returned")
                logger.info("OpenAI response received", response_repr=str(response)[:1000])

                # Safely extract generated content
                generated_content = ""
                try:
                    if hasattr(response, 'choices') and response.choices:
                        choice = response.choices[0]
                        # support different response shapes
                        if hasattr(choice, 'message') and getattr(choice.message, 'content', None):
                            generated_content = choice.message.content
                        else:
                            generated_content = getattr(choice, 'text', "") or ""
                except Exception as e:
                    print("[debug] Failed to extract generated content:", e)

                print("[debug] Generated content (preview):", (generated_content or "")[:400].replace('\n', ' '))
                parsed = self._parse_generated_test_case(generated_content or "", request)
                logger.info("Parsed test case", parsed_keys=list(parsed.keys()))
                return TestCase(**parsed)
            except Exception as e:
                # Print and log full traceback and error details
                print("[error] Exception in generate_test_case:", type(e).__name__, str(e))
                traceback_str = traceback.format_exc()
                print(traceback_str)
                logger.error("Failed to generate test case", error=str(e), traceback=traceback_str)
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
    
   
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for test case generation"""
        return (
            "You are an expert test case generator. Generate comprehensive, well-structured test cases based on the provided feature description and acceptance criteria.\n\n"
            "IMPORTANT: Reply with a single, valid JSON object ONLY (no surrounding markdown, explanation text, or backticks). "
            "The JSON must exactly follow the schema below and include all required fields. If a value is not available, "
            "provide a reasonable default (empty string, empty list, or N/A) rather than omitting the field.\n\n"
            "Required JSON structure:\n"
            "{\n"
            "  \"title\": string,\n"
            "  \"description\": string,\n"
            "  \"test_steps\": [\n"
            "    {\"step_number\": int, \"action\": string, \"expected_result\": string, \"test_data\": string}\n"
            "  ],\n"
            "  \"expected_result\": string,\n"
            "  \"preconditions\": string\n"
            "}\n\n"
            "Additional notes:\n"
            "- Make sure \"test_steps\" is an array with at least one step.\n"
            "- \"step_number\" should start at 1 and increment.\n"
            "- Keep text concise but complete. Use clear actions and expected results.\n"
            "- Do NOT include any commentary or explanation outside the JSON object.\n\n"
            "If you cannot produce a valid JSON object, return the empty JSON object: {}"
        )
    
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
                # Fallback: ask the model to extract/return ONLY the JSON object from its previous output
                parsed_data = None
                try:
                    extraction_prompt = (
                        "The assistant output below may contain a JSON object.\n"
                        "Please extract and return ONLY the JSON object. If no JSON can be found, return {}.\n\n" + generated_content
                    )
                    extraction_response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a JSON extractor. Respond ONLY with the JSON object found in the user content."},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        temperature=0.0,
                        top_p=1.0,
                        model=self.model
                    )

                    extracted_text = ""
                    if hasattr(extraction_response, 'choices') and extraction_response.choices:
                        ch = extraction_response.choices[0]
                        if hasattr(ch, 'message') and getattr(ch.message, 'content', None):
                            extracted_text = ch.message.content
                        else:
                            extracted_text = getattr(ch, 'text', '') or ''

                    json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                    if json_match:
                        parsed_data = json.loads(json_match.group())
                except Exception as ex:
                    logger.warning("Fallback JSON extraction failed", error=str(ex))

                if parsed_data is None:
                    # Final fallback to manual parsing
                    parsed_data = self._manual_parse_test_case(generated_content)
            
            # Convert test_steps to TestStep objects
            test_steps = []
            for step_data in parsed_data.get("test_steps", []):
                if 'test_data' not in step_data:
                    step_data['test_data'] = "N/A"
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
            updated_at=datetime.utcnow(),
            jira_issue_key=None
        )
