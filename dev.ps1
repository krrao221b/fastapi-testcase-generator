# Test Case Generator API - Development Helper Script
# PowerShell script for Windows development environment

param(
    [string]$Action = "help"
)

function Show-Help {
    Write-Host "🚀 Test Case Generator API - Development Helper" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  setup      - Initial project setup"
    Write-Host "  install    - Install dependencies"
    Write-Host "  run        - Start the development server"
    Write-Host "  test       - Run tests"
    Write-Host "  format     - Format code with black and isort"
    Write-Host "  lint       - Run linting with flake8"
    Write-Host "  clean      - Clean cache and temporary files"
    Write-Host "  help       - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\dev.ps1 setup"
    Write-Host "  .\dev.ps1 run"
    Write-Host "  .\dev.ps1 test"
}

function Invoke-Setup {
    Write-Host "🔧 Setting up Test Case Generator API..." -ForegroundColor Green
    
    # Check if Python is installed
    try {
        $pythonVersion = python --version
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
        exit 1
    }
    
    # Create virtual environment if it doesn't exist
    if (!(Test-Path "venv")) {
        Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
    
    # Install dependencies
    Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Create .env file if it doesn't exist
    if (!(Test-Path ".env")) {
        Write-Host "⚙️ Creating .env file from template..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "📝 Please edit .env file with your API keys and configuration." -ForegroundColor Cyan
    }
    
    # Create data directory
    if (!(Test-Path "data")) {
        New-Item -ItemType Directory -Path "data"
        Write-Host "📁 Created data directory" -ForegroundColor Green
    }
    
    Write-Host "✅ Setup complete! Run '.\dev.ps1 run' to start the server." -ForegroundColor Green
}

function Invoke-Install {
    Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed!" -ForegroundColor Green
}

function Invoke-Run {
    Write-Host "🚀 Starting Test Case Generator API..." -ForegroundColor Green
    Write-Host "📚 API Documentation will be available at: http://localhost:8000/api/v1/docs" -ForegroundColor Cyan
    Write-Host "🏥 Health Check: http://localhost:8000/api/v1/health" -ForegroundColor Cyan
    Write-Host "🔄 Use Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    python main.py
}

function Invoke-Test {
    Write-Host "🧪 Running tests..." -ForegroundColor Yellow
    pytest -v --tb=short
}

function Invoke-Format {
    Write-Host "✨ Formatting code..." -ForegroundColor Yellow
    
    Write-Host "  Running black..." -ForegroundColor Cyan
    black app/ tests/ main.py
    
    Write-Host "  Running isort..." -ForegroundColor Cyan
    isort app/ tests/ main.py
    
    Write-Host "✅ Code formatting complete!" -ForegroundColor Green
}

function Invoke-Lint {
    Write-Host "🔍 Running linting..." -ForegroundColor Yellow
    
    Write-Host "  Running flake8..." -ForegroundColor Cyan
    flake8 app/ tests/ main.py --max-line-length=88 --extend-ignore=E203,W503
    
    Write-Host "  Running mypy..." -ForegroundColor Cyan
    mypy app/ --ignore-missing-imports
    
    Write-Host "✅ Linting complete!" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "🧹 Cleaning cache and temporary files..." -ForegroundColor Yellow
    
    # Remove Python cache
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Remove test artifacts
    Remove-Item -Path ".pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path ".coverage" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "htmlcov" -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove build artifacts
    Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "✅ Cleanup complete!" -ForegroundColor Green
}

# Main execution
switch ($Action.ToLower()) {
    "setup" { Invoke-Setup }
    "install" { Invoke-Install }
    "run" { Invoke-Run }
    "test" { Invoke-Test }
    "format" { Invoke-Format }
    "lint" { Invoke-Lint }
    "clean" { Invoke-Clean }
    "help" { Show-Help }
    default { 
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Show-Help 
    }
}
