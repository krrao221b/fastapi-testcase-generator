from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test Case Generator API - Basic Test")

@app.get("/")
async def root():
    return {"message": "FastAPI backend is working!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running successfully"}

if __name__ == "__main__":
    print("🚀 Starting Test Case Generator API (Basic Test)...")
    print("📚 Open: http://localhost:4200")
    print("📚 Docs: http://localhost:4200/docs")
    print("🔄 Use Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=4200)
