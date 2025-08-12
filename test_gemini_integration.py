"""
Test Gemini Integration
"""
import asyncio
import sys
import os
sys.path.append('.')

async def test_gemini():
    print("🧪 Testing Gemini API Integration...")
    
    try:
        from app.repositories.implementations.gemini_service import GeminiService
        from app.models.schemas import GenerateTestCaseRequest, TestCasePriority
        
        # Create service
        service = GeminiService()
        print("✅ Gemini service created")
        
        # Create request
        request = GenerateTestCaseRequest(
            feature_description="User login functionality",
            acceptance_criteria="User can login with valid email and password",
            priority=TestCasePriority.HIGH,
            tags=["auth", "login"]
        )
        print("✅ Request created")
        
        # Test the API call
        print("🔄 Calling Gemini AI...")
        result = await service.generate_test_case(request)
        
        # Display results
        print("\n🎉 GEMINI SUCCESS!")
        print(f"📝 Title: {result.title}")
        print(f"📋 Description: {result.description[:100]}...")
        print(f"🔢 Test Steps: {len(result.test_steps)}")
        
        for i, step in enumerate(result.test_steps[:3], 1):
            print(f"   {i}. {step.action[:60]}...")
        
        print(f"🎯 Expected Result: {result.expected_result}")
        print(f"🏷️ Tags: {result.tags}")
        
        return True
        
    except Exception as e:
        print(f"❌ GEMINI FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_gemini())
