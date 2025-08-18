from typing import List, Optional
import structlog
from app.models.schemas import (
    TestCase, TestCaseCreate, TestCaseUpdate, 
    GenerateTestCaseRequest, GenerateNewTestCaseRequest, 
    GenerateTestCaseResponse, GenerateNewTestCaseResponse, 
    SearchSimilarRequest, SimilarTestCase
)
from app.repositories.interfaces.test_case_repository import ITestCaseRepository
from app.repositories.interfaces.memory_service import IMemoryService
from app.repositories.interfaces.ai_service import IAIService
from app.repositories.interfaces.jira_service import IJiraService
from app.repositories.interfaces.zephyr_service import IZephyrService

logger = structlog.get_logger()


class TestCaseService:
    """Business logic service for test case operations"""
    
    def __init__(
        self,
        test_case_repository: ITestCaseRepository,
        memory_service: IMemoryService,
        ai_service: IAIService,
        jira_service: IJiraService,
        zephyr_service: IZephyrService
    ):
        self.test_case_repository = test_case_repository
        self.memory_service = memory_service
        self.ai_service = ai_service
        self.jira_service = jira_service
        self.zephyr_service = zephyr_service
        
    async def get_jira_ticket(self, ticket_key: str) -> Optional[dict]:
        """Fetch JIRA ticket details by ticket key (e.g., PROJ-123)"""
        try:
            logger.info("Fetching JIRA ticket from JiraService", ticket_key=ticket_key)
            ticket_data = await self.jira_service.get_issue(ticket_key)
            if not ticket_data:
                return None

            # Safely extract fields with sensible defaults — JIRA adapter may not provide all keys
            description = ticket_data.get("description", "")
            acceptance_criteria = ticket_data.get("acceptance_criteria", "")
            additional_context = ticket_data.get("additional_context", "")
            priority = ticket_data.get("priority") or ticket_data.get("priority", "")
            tags = ticket_data.get("tags") or []
            jira_issue_key = ticket_data.get("jira_issue_key") or ticket_data.get("key") or ticket_key

            # Normalize priority to a string
            if hasattr(priority, "value"):
                priority_val = priority.value
            else:
                priority_val = str(priority) if priority is not None else ""

            combined_query = f"Feature: {description}\nAcceptance: {acceptance_criteria}"
            if tags:
                try:
                    combined_query += f"\nTags: {', '.join(tags)}"
                except Exception:
                    # Ensure tags are a list of strings
                    combined_query += f"\nTags: {tags}"
            combined_query += f"\nPriority: {priority_val}"

            similar_cases = await self.memory_service.search_similar(
                feature_description=combined_query,
                limit=5,
                threshold=0.7
            )

            # Convert SimilarTestCase objects to JSON-serializable dicts
            serializable_similar = []
            try:
                for sc in similar_cases or []:
                    test_case_obj = sc.test_case
                    # Pydantic models support model_dump / model_dump_json on v2; fall back to dict-like access
                    try:
                        tc_dict = test_case_obj.model_dump() if hasattr(test_case_obj, "model_dump") else test_case_obj.dict()
                    except Exception:
                        # Last resort: attempt to convert attributes
                        tc_dict = {k: getattr(test_case_obj, k) for k in dir(test_case_obj) if not k.startswith("_")}
                    serializable_similar.append({
                        "test_case": tc_dict,
                        "similarity_score": round(float(getattr(sc, "similarity_score", 0.0)), 4)
                    })
            except Exception:
                serializable_similar = []

            return {
                "jira_ticket_data": ticket_data,
                "similar_cases": serializable_similar,
            }
        except Exception as e:
            logger.error("Failed to fetch JIRA ticket from JiraService", ticket_key=ticket_key, error=str(e))
            return None
    
    async def generate_test_case(self, request: GenerateTestCaseRequest) -> GenerateTestCaseResponse:
        """Generate a new test case using AI and memory search"""
        try:
            logger.info("Starting test case generation", 
                       feature_description=request.feature_description[:100])

            # First, search for similar test cases
            # Use a combined text (feature + acceptance criteria + tags + priority) to match how embeddings are generated on store
            combined_query = f"Feature: {request.feature_description}\nAcceptance: {request.acceptance_criteria}"
            if request.tags:
                combined_query += f"\nTags: {', '.join(request.tags)}"
            combined_query += f"\nPriority: {request.priority.value if hasattr(request.priority, 'value') else request.priority}"

            similar_cases = await self.memory_service.search_similar(
                feature_description=combined_query,
                limit=5,
                threshold=0.7
            )

            # Log similarity scores for debugging (file and method context)
            try:
                sim_list = [(sc.test_case.id, round(sc.similarity_score, 4)) for sc in similar_cases]
            except Exception:
                sim_list = []
            logger.info("Similarity search results", file="test_case_service.py", method="generate_test_case", similarities=sim_list)
            print(f"[debug] test_case_service.py:generate_test_case - similarity scores: {sim_list}")

            # Strict duplicate check in DB (still performed) — but include the similar_cases in the response for visibility
            existing_case = await self.test_case_repository.find_by_feature_and_criteria(
                request.feature_description.strip(),
                request.acceptance_criteria.strip()
            )
            if existing_case:
                logger.info("Exact duplicate found in DB, returning existing test case", test_case_id=existing_case.id)
                gen_meta = {
                    "ai_model_used": "existing_duplicate",
                    "duplicate_detection": True,
                    "original_test_case_id": existing_case.id
                }
                # provide concise similar info
                gen_meta.update({
                    "most_similar_test_case_id": existing_case.id,
                    "most_similar_score": 1.0,
                    "similar_found": True
                })
                return GenerateTestCaseResponse(
                    test_case=existing_case,
                    similar_cases=similar_cases,
                    generation_metadata=gen_meta
                )

            # Provide a concise similar-case indicator for frontend (most similar)
            most_similar = None
            if similar_cases:
                most_similar = max(similar_cases, key=lambda s: s.similarity_score)
            if most_similar:
                logger.info("Most similar case found", test_case_id=most_similar.test_case.id, score=most_similar.similarity_score)

            
            # Check if we have a very similar test case (high similarity threshold)
            duplicate_threshold = 0.95
            potential_duplicate = None
            
            if similar_cases:
                for similar_case in similar_cases:
                    if similar_case.similarity_score >= duplicate_threshold:
                        # Check if feature description and acceptance criteria are very similar
                        if (similar_case.test_case.feature_description.strip().lower() == 
                            request.feature_description.strip().lower() and
                            similar_case.test_case.acceptance_criteria.strip().lower() == 
                            request.acceptance_criteria.strip().lower()):
                            potential_duplicate = similar_case.test_case
                            break
            
            # If duplicate found, return it instead of creating new one
            if potential_duplicate:
                logger.info("Found potential duplicate test case, returning existing one", 
                           existing_test_case_id=potential_duplicate.id)
                gen_meta = {
                    "ai_model_used": "existing_duplicate",
                    "similar_cases_found": len(similar_cases),
                    "generation_timestamp": potential_duplicate.created_at.isoformat(),
                    "duplicate_detection": True,
                    "original_test_case_id": potential_duplicate.id
                }
                # include most similar info
                gen_meta.update({
                    "most_similar_test_case_id": potential_duplicate.id,
                    "most_similar_score": 1.0,
                    "similar_found": True
                })
                return GenerateTestCaseResponse(
                    test_case=potential_duplicate,
                    similar_cases=similar_cases,
                    generation_metadata=gen_meta
                )
            
            # Decide whether to persist the generated test case based on similarity
            # If the caller set force_save=True, always save. Otherwise, skip saving when
            # a most_similar case exists with score >= settings.skip_store_if_similar_score
            from app.config.settings import settings

            should_force_save = getattr(request, "force_save", False)
            skip_store_threshold = float(settings.skip_store_if_similar_score or 0.0)

            # If a highly-similar case exists and caller did not force save, skip calling the AI
            # and return the existing similar case immediately. This avoids unnecessary LLM calls
            # when we already have a candidate to return to the frontend.
            if most_similar and not should_force_save and most_similar.similarity_score >= skip_store_threshold:
                logger.info("Skipping AI generation due to similar existing test case",
                           most_similar_id=most_similar.test_case.id, score=most_similar.similarity_score)

                gen_meta = {
                    "ai_model_used": "skipped_due_to_similarity",
                    "similar_cases_found": len(similar_cases),
                    "generation_timestamp": most_similar.test_case.created_at.isoformat() if getattr(most_similar.test_case, 'created_at', None) else None,
                    "duplicate_detection": True,
                    "is_new_generation": False,
                    "store_skipped": True,
                    "most_similar_test_case_id": most_similar.test_case.id,
                    "most_similar_score": most_similar.similarity_score
                }

                return GenerateTestCaseResponse(
                    test_case=most_similar.test_case,
                    similar_cases=similar_cases,
                    generation_metadata=gen_meta
                )

            # Generate new test case using AI (defensive: catch provider errors and return a fallback)
            try:
                ai_test_case = await self.ai_service.generate_test_case(request)
            except Exception as ai_exc:
                logger.error("AI generation failed, using fallback test case", error=str(ai_exc))
                # Build a minimal fallback TestCase similar to OpenAIService fallback
                from datetime import datetime
                from app.models.schemas import TestStep, TestCase as TC, TestCaseStatus
                ai_test_case = TC(
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

            # If parsed generation looks incomplete, attempt one retry
            def _is_incomplete(tc):
                try:
                    if not tc:
                        return True
                    if not getattr(tc, 'test_steps', None):
                        return True
                    if len(getattr(tc, 'test_steps', [])) == 0:
                        return True
                    if not getattr(tc, 'expected_result', None):
                        return True
                    return False
                except Exception:
                    return True

            if _is_incomplete(ai_test_case):
                logger.warning("AI generated test case looks incomplete, retrying once", file="test_case_service.py", method="generate_test_case")
                print("[warn] test_case_service.py:generate_test_case - AI result incomplete, retrying generation once")
                ai_test_case = await self.ai_service.generate_test_case(request)
            
            # Create the test case data dictionary
            test_case_data = ai_test_case.dict(exclude={"id", "created_at", "updated_at"})
            
            # Extract JIRA issue key from tags or use the provided jira_issue_key
            jira_issue_key = request.jira_issue_key
            
            # If no explicit JIRA issue key provided, look for it in tags
            if not jira_issue_key and request.tags:
                for tag in request.tags:
                    # Check if tag matches JIRA/SCRUM pattern (e.g., "SCRUM-22", "PROJ-123")
                    if tag and '-' in tag and tag.split('-')[0].isalpha() and tag.split('-')[1].isdigit():
                        jira_issue_key = tag
                        break
            
            # Add JIRA issue key if found
            if jira_issue_key:
                test_case_data["jira_issue_key"] = jira_issue_key
                logger.info("Assigned JIRA issue key to test case", jira_issue_key=jira_issue_key)
                
            # Decide whether to persist the generated test case based on similarity
            # If the caller set force_save=True, always save. Otherwise, skip saving when
            # a most_similar case exists with score >= settings.skip_store_if_similar_score
            from app.config.settings import settings

            should_force_save = getattr(request, "force_save", False)
            skip_store_threshold = float(settings.skip_store_if_similar_score or 0.0)

            if most_similar and not should_force_save and most_similar.similarity_score >= skip_store_threshold:
                # Do NOT save the generated case; return the generated content to caller but mark it as skipped
                logger.info("Skipping persistence of generated test case due to similarity to existing case",
                           most_similar_id=most_similar.test_case.id, score=most_similar.similarity_score)

                gen_meta = {
                    "ai_model_used": "openai",
                    "similar_cases_found": len(similar_cases),
                    "generation_timestamp": ai_test_case.created_at.isoformat(),
                    "duplicate_detection": True,
                    "is_new_generation": False,
                    "store_skipped": True,
                    "most_similar_test_case_id": most_similar.test_case.id,
                    "most_similar_score": most_similar.similarity_score
                }

                # Return the generated test case object (not persisted) so caller can preview or save explicitly
                return GenerateTestCaseResponse(
                    test_case=ai_test_case,
                    similar_cases=similar_cases,
                    generation_metadata=gen_meta
                )

            # Save the generated test case to database
            test_case_create = TestCaseCreate(**test_case_data)
            saved_test_case = await self.test_case_repository.create(test_case_create)
            
            # Store in vector database for future similarity searches
            await self.memory_service.store_test_case(saved_test_case)
            
            logger.info("Test case generated and saved successfully", 
                       test_case_id=saved_test_case.id)
            
            # Add concise similar info to generation metadata so frontend can easily check
            gen_meta = {
                "ai_model_used": "openai",
                "similar_cases_found": len(similar_cases),
                "generation_timestamp": saved_test_case.created_at.isoformat(),
                "duplicate_detection": False,
                "is_new_generation": True
            }
            if most_similar:
                gen_meta.update({
                    "most_similar_test_case_id": most_similar.test_case.id,
                    "most_similar_score": most_similar.similarity_score,
                    "similar_found": most_similar.similarity_score >= 0.8
                })

            return GenerateTestCaseResponse(
                test_case=saved_test_case,
                similar_cases=similar_cases,
                generation_metadata=gen_meta
            )
            
        except Exception as e:
            logger.error("Failed to generate test case", 
                        feature_description=request.feature_description, error=str(e))
            raise
    
    async def search_similar_test_cases(self, request: SearchSimilarRequest) -> List[SimilarTestCase]:
        """Search for similar test cases"""
        try:
            similar_cases = await self.memory_service.search_similar(
                feature_description=request.feature_description,
                limit=request.limit,
                threshold=request.similarity_threshold
            )
           
            logger.info("Similar test cases search completed",
                       feature_description=request.feature_description[:100],
                       results_count=len(similar_cases))
           
            return similar_cases
           
        except Exception as e:
            logger.error("Failed to search similar test cases",
                        feature_description=request.feature_description, error=str(e))
            raise

    async def generate_new_test_case(self, request: GenerateNewTestCaseRequest) -> GenerateNewTestCaseResponse:
        """Generate a new test case using AI and memory search"""
        try:
            # Generate new test case using AI
            ai_test_case = await self.ai_service.generate_new_test_case(request)

            # If parsed generation looks incomplete, attempt one retry
            def _is_incomplete(tc):
                try:
                    if not tc:
                        return True
                    if not getattr(tc, 'test_steps', None):
                        return True
                    if len(getattr(tc, 'test_steps', [])) == 0:
                        return True
                    if not getattr(tc, 'expected_result', None):
                        return True
                    return False
                except Exception:
                    return True

            if _is_incomplete(ai_test_case):
                logger.warning("AI generated test case looks incomplete, retrying once", file="test_case_service.py",
                               method="generate_test_case")
                print("[warn] test_case_service.py:generate_test_case - AI result incomplete, retrying generation once")
                ai_test_case = await self.ai_service.generate_test_case(request)

            # Create the test case data dictionary
            test_case_data = ai_test_case.dict(exclude={"id", "created_at", "updated_at"})

            # Extract JIRA issue key from tags or use the provided jira_issue_key
            jira_issue_key = request.jira_issue_key

            # If no explicit JIRA issue key provided, look for it in tags
            if not jira_issue_key and request.tags:
                for tag in request.tags:
                    # Check if tag matches JIRA/SCRUM pattern (e.g., "SCRUM-22", "PROJ-123")
                    if tag and '-' in tag and tag.split('-')[0].isalpha() and tag.split('-')[1].isdigit():
                        jira_issue_key = tag
                        break

            # Add JIRA issue key if found
            if jira_issue_key:
                test_case_data["jira_issue_key"] = jira_issue_key
                logger.info("Assigned JIRA issue key to test case", jira_issue_key=jira_issue_key)

            # Persist the newly generated test case
            logger.info("Saving newly generated test case", feature_description=request.feature_description[:100])
            test_case_create = TestCaseCreate(**test_case_data)
            saved_test_case = await self.test_case_repository.create(test_case_create)

            # Store in vector database for future similarity searches
            try:
                await self.memory_service.store_test_case(saved_test_case)
            except Exception as mem_exc:
                logger.warning("Failed to store test case embedding (generate_new)", error=str(mem_exc))

            gen_meta = {
                "ai_model_used": "openai",
                "generation_timestamp": saved_test_case.created_at.isoformat(),
                "duplicate_detection": False,
                "is_new_generation": True,
                "store_skipped": False,
            }
            return GenerateNewTestCaseResponse(
                test_case=saved_test_case,
                generation_metadata=gen_meta,
                message=f"Generated and saved new test case with ID {saved_test_case.id}",
            )
        except Exception as e:
            logger.error("Failed to generate new test case", error=str(e))
            raise

    async def get_test_case(self, test_case_id: int) -> Optional[TestCase]:
        """Get a test case by ID"""
        return await self.test_case_repository.get_by_id(test_case_id)
    
    async def get_all_test_cases(self, skip: int = 0, limit: int = 100) -> List[TestCase]:
        """Get all test cases with pagination"""
        return await self.test_case_repository.get_all(skip=skip, limit=limit)
    
    async def update_test_case(self, test_case_id: int, update_data: TestCaseUpdate) -> Optional[TestCase]:
        """Update an existing test case"""
        try:
            updated_test_case = await self.test_case_repository.update(test_case_id, update_data)
            
            if updated_test_case:
                # Update the vector database embedding
                await self.memory_service.update_test_case_embedding(updated_test_case)
                
                logger.info("Test case updated successfully", test_case_id=test_case_id)
            
            return updated_test_case
            
        except Exception as e:
            logger.error("Failed to update test case", test_case_id=test_case_id, error=str(e))
            raise
    
    async def delete_test_case(self, test_case_id: int) -> bool:
        """Delete a test case"""
        try:
            # Delete from vector database first
            await self.memory_service.delete_test_case_embedding(test_case_id)
            
            # Delete from main database
            success = await self.test_case_repository.delete(test_case_id)
            
            if success:
                logger.info("Test case deleted successfully", test_case_id=test_case_id)
            
            return success
            
        except Exception as e:
            logger.error("Failed to delete test case", test_case_id=test_case_id, error=str(e))
            raise
    
    async def integrate_with_jira(self, test_case_id: int, create_issue: bool = False) -> Optional[str]:
        """Integrate test case with JIRA"""
        try:
            test_case = await self.test_case_repository.get_by_id(test_case_id)
            if not test_case:
                return None
            
            if create_issue:
                # Create JIRA issue
                issue_key = await self.jira_service.create_issue(
                    title=test_case.title,
                    description=f"{test_case.description}\n\nFeature: {test_case.feature_description}\n\nAcceptance Criteria: {test_case.acceptance_criteria}"
                )
                
                if issue_key:
                    # Update test case with JIRA issue key
                    await self.test_case_repository.update(
                        test_case_id, 
                        TestCaseUpdate(jira_issue_key=issue_key)
                    )
                    
                    logger.info("Test case integrated with JIRA", 
                              test_case_id=test_case_id, issue_key=issue_key)
                
                return issue_key
            
            return test_case.jira_issue_key
            
        except Exception as e:
            logger.error("Failed to integrate with JIRA", 
                        test_case_id=test_case_id, error=str(e))
            raise
    
    async def improve_test_case_with_ai(self, test_case_id: int, feedback: str) -> Optional[TestCase]:
        """Improve a test case using AI based on feedback"""
        try:
            test_case = await self.test_case_repository.get_by_id(test_case_id)
            if not test_case:
                return None
            
            # Use AI to improve the test case
            improved_test_case = await self.ai_service.improve_test_case(test_case, feedback)
            
            # Update the test case in database
            update_data = TestCaseUpdate(
                title=improved_test_case.title,
                description=improved_test_case.description,
                test_steps=improved_test_case.test_steps,
                expected_result=improved_test_case.expected_result,
                preconditions=improved_test_case.preconditions
            )
            
            updated_test_case = await self.test_case_repository.update(test_case_id, update_data)
            
            if updated_test_case:
                # Update vector database
                await self.memory_service.update_test_case_embedding(updated_test_case)
                
                logger.info("Test case improved with AI", test_case_id=test_case_id)
            
            return updated_test_case
            
        except Exception as e:
            logger.error("Failed to improve test case with AI", 
                        test_case_id=test_case_id, error=str(e))
            raise
