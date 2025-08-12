# Test Case Generator API

Enterprise-grade FastAPI backend for AI-powered test case generation with JIRA and Zephyr integration.

## 🏗️ Architecture

This project follows **Clean Architecture** principles with clear separation of concerns:

```
app/
├── api/                    # 🌐 Presentation Layer (Controllers)
│   └── routes/            # FastAPI route handlers
├── services/              # 💼 Business Logic Layer
├── repositories/          # 🗄️ Data Access Layer
│   ├── interfaces/        # Abstract interfaces
│   └── implementations/   # Concrete implementations
├── models/               # 📋 Data Models
├── core/                 # ⚙️ Infrastructure
└── config/               # 🔧 Configuration
```

## 🚀 Features

- **AI-Powered Generation**: Uses Google Gemini for intelligent test case creation
- **Memory Search**: ChromaDB for semantic similarity matching
- **JIRA Integration**: Seamless integration with Atlassian JIRA
- **Zephyr Integration**: Test management with Zephyr Scale
- **Clean Architecture**: SOLID principles, dependency injection
- **Type Safety**: Full Pydantic validation and type hints
- **Structured Logging**: JSON logging with correlation IDs
- **Health Monitoring**: Built-in health and readiness checks

## 🛠️ Tech Stack

| Component      | Technology                     |
| -------------- | ------------------------------ |
| **Framework**  | FastAPI 0.104+                 |
| **Database**   | SQLAlchemy + SQLite/PostgreSQL |
| **Vector DB**  | ChromaDB                       |
| **AI Service** | Google Gemini                  |
| **Validation** | Pydantic V2                    |
| **Logging**    | Structlog                      |
| **Testing**    | Pytest + AsyncIO               |

## 📦 Installation

1. **Clone and setup**:

```bash
git clone <repository>
cd FastApi
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure environment**:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Run the application**:

```bash
python main.py
```

## 🔧 Configuration

Required environment variables:

```env
# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Integrations
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your_jira_username
JIRA_API_TOKEN=your_jira_api_token

ZEPHYR_BASE_URL=https://your-zephyr-instance.com
ZEPHYR_ACCESS_KEY=your_zephyr_access_key
ZEPHYR_SECRET_KEY=your_zephyr_secret_key
```

## 📚 API Documentation

Once running, visit:

- **Interactive Docs**: http://localhost:4200/api/v1/docs
- **ReDoc**: http://localhost:4200/api/v1/redoc

## 🏃‍♂️ Quick Start

### Generate a Test Case

```python
import httpx

async def generate_test_case():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:4200/api/v1/test-cases/generate",
            json={
                "feature_description": "User login functionality",
                "acceptance_criteria": "User should be able to login with valid credentials",
                "priority": "high",
                "tags": ["authentication", "security"]
            }
        )
        return response.json()
```

### Search Similar Test Cases

```python
async def search_similar():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:4200/api/v1/test-cases/search-similar",
            json={
                "feature_description": "User authentication",
                "limit": 5,
                "similarity_threshold": 0.7
            }
        )
        return response.json()
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_services.py
```

## 🏗️ Architecture Details

### SOLID Principles Implementation

1. **Single Responsibility**: Each service has one clear purpose
2. **Open/Closed**: Extensible through interfaces
3. **Liskov Substitution**: Implementations are interchangeable
4. **Interface Segregation**: Small, focused interfaces
5. **Dependency Inversion**: Depends on abstractions, not concretions

### Design Patterns Used

- **Repository Pattern**: Data access abstraction
- **Dependency Injection**: Loose coupling between components
- **Factory Pattern**: Service creation and configuration
- **Strategy Pattern**: Multiple AI/integration providers

### Error Handling

- Global exception handlers
- Structured error responses
- Detailed logging with correlation IDs
- Graceful degradation for external services

## 🔄 Development Workflow

1. **Add new feature**:

   - Define interface in `repositories/interfaces/`
   - Implement in `repositories/implementations/`
   - Add business logic to `services/`
   - Create API endpoint in `api/routes/`
   - Write tests

2. **Code quality**:

```bash
# Format code
black app/
isort app/

# Type checking
mypy app/

# Linting
flake8 app/
```

## 🌟 Key Benefits

- **Maintainable**: Clear separation of concerns
- **Testable**: Interface-based design enables easy mocking
- **Scalable**: Modular architecture supports growth
- **Reliable**: Comprehensive error handling and logging
- **Flexible**: Easy to swap implementations (databases, AI providers, etc.)

## 🤝 Team Integration

Perfect for team development:

- **Team A**: AI prompt design (`repositories/implementations/gemini_service.py`)
- **Team B**: API endpoints (`api/routes/`)
- **Team C**: Frontend integration (consuming the APIs)
- **Team D**: Database and memory operations (`repositories/implementations/`)

## 📈 Production Readiness

- Environment-based configuration
- Health check endpoints
- Structured logging
- Error monitoring ready
- Database migrations support
- Docker deployment ready

## 🔮 Future Enhancements

- [ ] Authentication & Authorization
- [ ] File upload support
- [ ] Batch operations
- [ ] Webhook integrations
- [ ] Advanced analytics
- [ ] Custom AI model fine-tuning

---

Built with ❤️ by Team Chai using FastAPI and modern Python best practices.
