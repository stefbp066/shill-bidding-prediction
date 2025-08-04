# Makefile for Shill Bidding Model Monitoring
# Best practices for development, testing, and deployment

.PHONY: help install test test-unit test-integration lint format clean deploy destroy deploy-cloud destroy-cloud
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := python -m pytest --import-mode=importlib
BLACK := black
FLAKE8 := flake8
ISORT := isort
MYPY := mypy
BANDIT := bandit

# Directories
API_DIR := api
TESTS_DIR := tests
TERRAFORM_DIR := terraform

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Shill Bidding Model Monitoring - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install all dependencies
	@echo "$(YELLOW)Installing Python dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r $(API_DIR)/requirements.txt
	$(PIP) install black flake8 isort mypy pytest pytest-cov pytest-mock bandit
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install -r $(API_DIR)/requirements.txt
	$(PIP) install black flake8 isort mypy pytest pytest-cov pytest-mock bandit pre-commit
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

format: ## Format code with Black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(BLACK) $(API_DIR)/ $(TESTS_DIR)/
	$(ISORT) $(API_DIR)/ $(TESTS_DIR)/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run linting checks
	@echo "$(YELLOW)Running linting checks...$(NC)"
	$(BLACK) --check --diff $(API_DIR)/ $(TESTS_DIR)/
	$(ISORT) --check-only --diff $(API_DIR)/ $(TESTS_DIR)/
	$(FLAKE8) $(API_DIR)/ $(TESTS_DIR)/ --max-line-length=88 --extend-ignore=E203,W503
	$(MYPY) $(API_DIR)/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting passed$(NC)"

security: ## Run security scan with bandit
	@echo "$(YELLOW)Running security scan...$(NC)"
	$(BANDIT) -r $(API_DIR)/ -f json -o bandit-report.json || true
	@echo "$(GREEN)✓ Security scan completed$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running all tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/ -v --cov=$(API_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/unit/ -v --cov=$(API_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/integration/ -v --cov=$(API_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Integration tests completed$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTEST) $(TESTS_DIR)/ -v --cov=$(API_DIR) --cov-report=html --cov-report=xml --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated$(NC)"

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "bandit-report.json" -delete
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

# Terraform commands
terraform-init: ## Initialize Terraform
	@echo "$(YELLOW)Initializing Terraform...$(NC)"
	cd $(TERRAFORM_DIR) && terraform init
	@echo "$(GREEN)✓ Terraform initialized$(NC)"

terraform-plan: ## Plan Terraform deployment
	@echo "$(YELLOW)Planning Terraform deployment...$(NC)"
	cd $(TERRAFORM_DIR) && terraform plan
	@echo "$(GREEN)✓ Terraform plan completed$(NC)"

terraform-apply: ## Apply Terraform configuration
	@echo "$(YELLOW)Applying Terraform configuration...$(NC)"
	cd $(TERRAFORM_DIR) && terraform apply -auto-approve
	@echo "$(GREEN)✓ Terraform apply completed$(NC)"

terraform-destroy: ## Destroy Terraform infrastructure
	@echo "$(YELLOW)Destroying Terraform infrastructure...$(NC)"
	cd $(TERRAFORM_DIR) && terraform destroy -auto-approve
	@echo "$(GREEN)✓ Terraform destroy completed$(NC)"

terraform-output: ## Show Terraform outputs
	@echo "$(YELLOW)Showing Terraform outputs...$(NC)"
	cd $(TERRAFORM_DIR) && terraform output
	@echo "$(GREEN)✓ Terraform outputs displayed$(NC)"

# Deployment commands
deploy: test lint security terraform-apply ## Deploy with full validation
	@echo "$(GREEN)✓ Deployment completed successfully$(NC)"

deploy-staging: test lint security terraform-plan ## Deploy to staging
	@echo "$(GREEN)✓ Staging deployment ready$(NC)"

deploy-production: test lint security terraform-apply ## Deploy to production
	@echo "$(GREEN)✓ Production deployment completed$(NC)"

deploy-cloud: ## Deploy to AWS cloud with Docker
	@echo "$(YELLOW)Deploying to AWS cloud...$(NC)"
	./scripts/deploy.sh
	@echo "$(GREEN)✓ Cloud deployment completed$(NC)"

destroy-cloud: ## Destroy AWS infrastructure
	@echo "$(YELLOW)Destroying AWS infrastructure...$(NC)"
	cd terraform && terraform destroy -auto-approve
	@echo "$(GREEN)✓ AWS infrastructure destroyed$(NC)"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	docker build -t shill-bidding-api $(API_DIR)/
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(NC)"
	docker run -p 8000:8000 shill-bidding-api
	@echo "$(GREEN)✓ Docker container running$(NC)"

docker-test: ## Run tests in Docker
	@echo "$(YELLOW)Running tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app -w /app python:3.9-slim bash -c "pip install pytest && pytest tests/"
	@echo "$(GREEN)✓ Docker tests completed$(NC)"

# Pre-commit hooks
install-hooks: ## Install pre-commit hooks
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

run-hooks: ## Run pre-commit hooks
	@echo "$(YELLOW)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit hooks completed$(NC)"

# Development workflow
dev-setup: install-dev install-hooks ## Set up development environment
	@echo "$(GREEN)✓ Development environment ready$(NC)"

dev-test: format lint test ## Run full development test suite
	@echo "$(GREEN)✓ Development tests completed$(NC)"

# CI/CD pipeline commands
ci-lint: lint ## CI: Run linting
	@echo "$(GREEN)✓ CI linting completed$(NC)"

ci-test: test-coverage ## CI: Run tests with coverage
	@echo "$(GREEN)✓ CI tests completed$(NC)"

ci-security: security ## CI: Run security scan
	@echo "$(GREEN)✓ CI security scan completed$(NC)"

ci-terraform: terraform-init terraform-plan ## CI: Run Terraform plan
	@echo "$(GREEN)✓ CI Terraform plan completed$(NC)"

# Utility commands
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "  Python version: $(shell python3 --version)"
	@echo "  Terraform version: $(shell terraform --version | head -n1)"
	@echo "  Docker version: $(shell docker --version)"
	@echo "  AWS CLI version: $(shell aws --version 2>/dev/null || echo 'Not installed')"

logs: ## Show application logs
	@echo "$(YELLOW)Showing application logs...$(NC)"
	@if [ -f "logs/app.log" ]; then tail -f logs/app.log; else echo "No log file found"; fi

monitor: ## Monitor application health
	@echo "$(YELLOW)Monitoring application health...$(NC)"
	@curl -s http://localhost:8000/health || echo "Application not running"

# Documentation
docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@mkdir -p docs
	@echo "# Shill Bidding Model Monitoring" > docs/README.md
	@echo "Generated documentation in docs/" >> docs/README.md
	@echo "$(GREEN)✓ Documentation generated$(NC)"

# Backup and restore
backup: ## Create backup of current state
	@echo "$(YELLOW)Creating backup...$(NC)"
	@mkdir -p backups
	@tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz --exclude=node_modules --exclude=.git .
	@echo "$(GREEN)✓ Backup created$(NC)"

restore: ## Restore from latest backup
	@echo "$(YELLOW)Restoring from backup...$(NC)"
	@ls -t backups/backup-*.tar.gz | head -1 | xargs -I {} tar -xzf {} --strip-components=1
	@echo "$(GREEN)✓ Restore completed$(NC)"

# Performance testing
benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	@python3 -c "import time; import requests; import json; data = {'auction_id': 'benchmark_test', 'bidder_tendency': 0.5, 'bidding_ratio': 0.6, 'successive_outbidding': 0.4, 'last_bidding': 0.3, 'auction_bids': 0.7, 'starting_price_average': 0.8, 'early_bidding': 0.2, 'winning_ratio': 0.9}; times = []; [times.append(time.time() - start) for i in range(10) if (start := time.time()) and (requests.post('http://localhost:8000/predict', json=data) or True)]; avg_time = sum(times) / len(times) if times else 0; print(f'Average response time: {avg_time:.3f}s'); print(f'Min response time: {min(times):.3f}s' if times else 'N/A'); print(f'Max response time: {max(times):.3f}s' if times else 'N/A')" || echo "API not running"
	@echo "$(GREEN)✓ Benchmark completed$(NC)"

# Quick commands for common tasks
quick-test: ## Quick test run
	@echo "$(YELLOW)Running quick tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/unit/ -v --tb=short
	@echo "$(GREEN)✓ Quick tests completed$(NC)"

quick-lint: ## Quick lint check
	@echo "$(YELLOW)Running quick lint...$(NC)"
	$(BLACK) --check $(API_DIR)/
	$(FLAKE8) $(API_DIR)/ --max-line-length=88
	@echo "$(GREEN)✓ Quick lint completed$(NC)"

# Show help for specific category
help-dev: ## Show development commands
	@echo "$(BLUE)Development Commands:$(NC)"
	@echo "  install-dev    - Install development dependencies"
	@echo "  dev-setup      - Set up development environment"
	@echo "  dev-test       - Run full development test suite"
	@echo "  format         - Format code"
	@echo "  lint           - Run linting checks"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"

help-deploy: ## Show deployment commands
	@echo "$(BLUE)Deployment Commands:$(NC)"
	@echo "  deploy         - Deploy with full validation"
	@echo "  deploy-staging - Deploy to staging"
	@echo "  deploy-production - Deploy to production"
	@echo "  terraform-init - Initialize Terraform"
	@echo "  terraform-plan - Plan Terraform deployment"
	@echo "  terraform-apply - Apply Terraform configuration"
	@echo "  terraform-destroy - Destroy Terraform infrastructure"

help-docker: ## Show Docker commands
	@echo "$(BLUE)Docker Commands:$(NC)"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-test    - Run tests in Docker"

help-ci: ## Show CI/CD commands
	@echo "$(BLUE)CI/CD Commands:$(NC)"
	@echo "  ci-lint        - CI: Run linting"
	@echo "  ci-test        - CI: Run tests with coverage"
	@echo "  ci-security    - CI: Run security scan"
	@echo "  ci-terraform   - CI: Run Terraform plan"
