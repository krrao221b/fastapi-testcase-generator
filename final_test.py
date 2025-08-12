"""
Quick Test for Gemini Integration
"""
import asyncio
import httpx

async def test_gemini_api_endpoint():
    """Test the cleaned up Gemini API endpoint"""
    print("🧪 Testing Gemini API via HTTP endpoint...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:4200/api/v1/test-cases/generate",
                json={
                    "feature_description": "User profile management",
                    "acceptance_criteria": "Users can update their profile information",
                    "priority": "medium",
                    "tags": ["profile", "user-management"]
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ SUCCESS! API is working")
                print(f"📝 Generated Title: {data['test_case']['title']}")
                print(f"🔢 Test Steps: {len(data['test_case']['test_steps'])}")
                print(f"🏷️ Tags: {data['test_case']['tags']}")
                return True
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        return False

async def test_health_check():
    """Test health check endpoint"""
    print("🏥 Testing health check...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:4200/api/v1/health")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Health Check: PASSED")
                print(f"🔧 Gemini Status: {data['checks']['gemini']}")
                return True
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Gemini Integration Tests...\n")
    
    # Test health check first
    health_ok = await test_health_check()
    print()
    
    # Test API endpoint
    api_ok = await test_gemini_api_endpoint()
    print()
    
    if health_ok and api_ok:
        print("🎉 ALL TESTS PASSED! Gemini integration is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    asyncio.run(main())
