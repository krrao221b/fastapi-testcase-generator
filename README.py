"""
Test Case Generator API

Enterprise-grade FastAPI backend for AI-powered test case generation with JIRA and Zephyr integration.

Architecture Overview:
- Clean Architecture with clear separation of concerns
- Repository pattern for data access
- Dependency Injection for loose coupling
- Interface-based design following SOLID principles

Key Features:
- AI-powered test case generation using OpenAI
- Semantic similarity search using ChromaDB
- JIRA integration for issue management
- Zephyr integration for test management
- Structured logging and monitoring
- Comprehensive error handling

Usage:
1. Copy .env.example to .env and configure your API keys
2. Install dependencies: pip install -r requirements.txt
3. Run the application: python main.py
4. Access API docs at: http://localhost:8000/api/v1/docs

API Endpoints:
- POST /api/v1/test-cases/generate - Generate new test case
- POST /api/v1/test-cases/search-similar - Find similar test cases
- GET /api/v1/test-cases/{id} - Get test case by ID
- PUT /api/v1/test-cases/{id} - Update test case
- DELETE /api/v1/test-cases/{id} - Delete test case
- POST /api/v1/integrations/jira/{id} - Integrate with JIRA
- POST /api/v1/integrations/zephyr/{id} - Integrate with Zephyr
- GET /api/v1/health - Health check

Architecture Components:

1. Controllers (app/api/routes/):
   - Handle HTTP requests and responses
   - Input validation using Pydantic
   - Route definition and documentation

2. Services (app/services/):
   - Business logic implementation
   - Orchestrates repository operations
   - Handles complex workflows

3. Repositories (app/repositories/):
   - Data access layer
   - Interface-based design for testability
   - Multiple implementations (SQL, ChromaDB, etc.)

4. Models (app/models/):
   - Pydantic schemas for request/response
   - SQLAlchemy models for database

5. Core (app/core/):
   - Database configuration
   - Dependency injection
   - Application infrastructure

6. Configuration (app/config/):
   - Environment-based settings
   - Type-safe configuration management
"""

__version__ = "1.0.0"
__author__ = "Team Chai"
__description__ = "AI-powered test case generation and management system"
