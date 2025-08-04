![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)
![Terraform](https://img.shields.io/badge/Terraform-✓-purple.svg)
![AWS](https://img.shields.io/badge/AWS-EC2-orange.svg)
[![Prefect](https://img.shields.io/badge/Orchestrator-Prefect-4B4FEB?logo=prefect&logoColor=white)](https://www.prefect.io/)

# Shill Bidding Model Monitoring System

A comprehensive ML model monitoring system for detecting shill bidding in auctions, featuring automated CI/CD pipeline, testing, and infrastructure as code.

**Access the deployed application:**
- **API Endpoint**: http://13.49.246.221:8000
- **API Documentation**: http://13.49.246.221:8000/docs
- **Grafana Dashboard**: http://13.49.246.221:3000
<img width="1438" height="639" alt="image" src="https://github.com/user-attachments/assets/cd549788-2bb8-4b77-bd3e-9232abebe67b" />
- **Prometheus**: http://13.49.246.221:9090

Test a prediction:
```bash
curl -X POST "http://13.49.246.221:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "auction_id": "demo_1",
    "bidder_tendency": 0.5,
    "bidding_ratio": 0.3,
    "successive_outbidding": 0.2,
    "last_bidding": 0.1,
    "auction_bids": 10,
    "starting_price_average": 100.0,
    "early_bidding": 0.4,
    "winning_ratio": 0.6,
    "auction_duration": 7
  }'
```

## 📚 Source

This project uses the **Shill Bidding Dataset** from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/562/shill+bidding+dataset). The dataset simulates online auction behavior and contains features extracted from bidding activity that may indicate fraudulent (i.e., shill bidding) patterns. Each record represents a bidder's behavior within an auction, with labeled classes distinguishing between suspicious and legitimate bidding behavior.

---

## 🚨 Problem

The goal of this project is to build a robust machine learning pipeline that can **automatically flag suspicious bidding behavior** based on historical auction data. Specifically, the model classifies each bidder within a new auction as either potentially fraudulent (shill bidder) or legitimate.

This task presents several practical **challenges** during the model development and training phases:

- **Imbalanced Target Classes**
  The dataset is highly skewed, with far fewer shill bidding cases than legitimate ones. This imbalance can bias models toward predicting the majority class. To address this, the pipeline uses **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples of the minority class during training, improving the model's ability to detect fraud.

- **Data Leakage Due to Shared Auction Context**
  Because multiple bidders participate in the same auction, there's a risk that data from one bidder might inadvertently reveal information about another. To prevent this form of leakage, **auction-level splitting** is applied. This ensures that all records from a given auction are placed entirely within either the training, validation, or test set—**never across multiple sets**.

- **Feature Engineering Pitfalls from Global Standardization**
  Standardizing features using global statistics (e.g., mean, std) can leak information across auctions, especially since auctions vary significantly in size, duration, and pricing. To maintain auction-level independence, the pipeline uses **pre-saved auction-specific statistics** for feature standardization. This simulates a real-time production scenario where only local auction context is available.

This problem setting attemtps to mirror real-world fraud detection systems.

## 🚀 Features

- **ML Model API**: FastAPI-based prediction service with drift detection
- **Monitoring Stack**: Grafana + Prometheus for real-time metrics visualization. However, **there is no retraining flow in case monitoring metrics are violated yet**.
- **Infrastructure as Code**: Terraform for AWS infrastructure deployment and versioning
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment
- **Testing**: Unit and integration tests with coverage reporting
- **Code Quality**: Linting, formatting, and security scanning
- **Best Practices**: Pre-commit hooks, Makefile, and comprehensive documentation

## 📋 Grading Requirements
### ☁️ Cloud & Infrastructure
- Remote storage (e.g., S3) used for storing artifacts
- Infrastructure-as-code via **Terraform**
- Project is cloud-ready and can be deployed end-to-end remotely

### 🔬 Experiment Tracking
- **MLflow** is used to:
  - Log training runs, parameters, and metrics
  - Track artifacts and models
  - Register models for versioning and reproducibility

### 🧪 Model Registry
- Registered models are version-controlled in **MLflow Model Registry**
- Staged transitions (e.g., `Staging`, `Production`) supported for testing and deployment flows

### ⚙️ Workflow Orchestration
- [![Prefect]](https://www.prefect.io/) handles the complete pipeline:
  - Data ingestion
  - Preprocessing
  - Training
  - Evaluation
  - Model registration
- Modular task structure for easy maintenance and expansion

### 📦 Deployment
- Model is **containerized with Docker**
- Includes a standalone inference script and API-ready setup
- Image can be deployed to cloud or on-prem environments

### 📈 Monitoring
- Model performance metrics are calculated and logged post-deployment
- Monitoring in place, though **retraining is not yet triggered automatically** from alerts or thresholds

### 🔁 Reproducibility
- All dependencies are version-pinned
- Environment and workflow reproducible locally or on the cloud

### Best practice checks:
✅ **Unit Tests** (1 point) - Comprehensive unit tests for model and API

✅ **Integration Tests** (1 point) - Full API integration testing

✅ **Linter and Code Formatter** (1 point) - Black, Flake8, isort, MyPy

✅ **Makefile** (1 point) - Comprehensive development and deployment commands

✅ **Pre-commit Hooks** (1 point) - Automated code quality checks

✅ **CI/CD Pipeline** (2 points) - GitHub Actions with multi-stage deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub Repo   │    │  GitHub Actions │    │   AWS Cloud     │
│                 │    │                 │    │                 │
│ • Source Code   │───▶│ • Lint & Format │───▶│ • EC2 Instance  │
│ • Tests         │    │ • Unit Tests    │    │ • S3 Bucket     │
│ • Terraform     │    │ • Integration   │    │ • VPC & Security│
│ • CI/CD Config  │    │ • Security Scan │    │ • Monitoring    │
└─────────────────┘    │ • Deploy        │    └─────────────────┘
                       └─────────────────┘
```

## 🛠️ Complete Setup Instructions

### Prerequisites

- **Python 3.9+** with virtual environment support
- **Docker & Docker Compose** for containerization
- **Terraform 1.0+** for infrastructure management
- **AWS CLI** configured with appropriate permissions
- **Git** for version control
- **SSH key pair** for EC2 access

### Step 1: Local Development Setup

1. **Clone and setup the repository**
   ```bash
   git clone https://github.com/stefbp066/shill-bidding-prediction.git
   cd shill-bidding-prediction

   # Create and activate virtual environment
   python -m venv mlops-env
   source mlops-env/bin/activate  # On Windows: mlops-env\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   # Install development dependencies
   pip install -r requirements.txt

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Train and register the ML model**
   ```bash
   # Train the model and register it with MLflow
   python src/register_model_for_api.py
   ```

4. **Test the setup locally**
   ```bash
   # Run all tests
   make test

   # Start local development server
   make dev-test
   ```

### Step 2: AWS Infrastructure Setup

1. **Configure AWS credentials**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, Default region (eu-north-1), and output format (json)
   ```

2. **Create SSH key pair for EC2 access**
   ```bash
   # Generate SSH key pair
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/shill-bidding-key -N ""

   # Import to AWS (replace with your key name)
   aws ec2 import-key-pair \
     --key-name shill-bidding-key \
     --public-key-material fileb://~/.ssh/shill-bidding-key.pub
   ```

3. **Deploy infrastructure with Terraform**
   ```bash
   cd terraform

   # Initialize Terraform
   terraform init

   # Plan the deployment
   terraform plan -var="key_pair_name=shill-bidding-key"

   # Apply the infrastructure
   terraform apply -var="key_pair_name=shill-bidding-key" -auto-approve
   ```

4. **Get the EC2 instance details**
   ```bash
   # Get the public IP address
   aws ec2 describe-instances \
     --filters "Name=tag:Name,Values=shill-bidding-app" \
     --query 'Reservations[].Instances[].PublicIpAddress' \
     --output text
   ```

### Step 3: Application Deployment

1. **Create deployment directory on EC2**
   ```bash
   # SSH to the instance (replace with your EC2 IP)
   ssh -i ~/.ssh/shill-bidding-key ec2-user@YOUR_EC2_IP

   # Create application directory
   sudo mkdir -p /opt/shill-bidding-monitoring
   sudo chown ec2-user:ec2-user /opt/shill-bidding-monitoring
   ```

2. **Upload application files**
   ```bash
   # From your local machine, upload the application files
   scp -i ~/.ssh/shill-bidding-key -r api ec2-user@YOUR_EC2_IP:/opt/shill-bidding-monitoring/
   scp -i ~/.ssh/shill-bidding-key -r monitoring ec2-user@YOUR_EC2_IP:/opt/shill-bidding-monitoring/
   scp -i ~/.ssh/shill-bidding-key -r data ec2-user@YOUR_EC2_IP:/opt/shill-bidding-monitoring/

   # Upload only the registered model (**not entire mlruns directory**)
   scp -i ~/.ssh/shill-bidding-key -r mlruns/108907814007183194/models/m-fe86ad04e2c14a1bafe22c853dae45e0 \
     ec2-user@YOUR_EC2_IP:/opt/shill-bidding-monitoring/mlruns/108907814007183194/models/

   # Upload docker-compose file
   scp -i ~/.ssh/shill-bidding-key docker-compose.yml ec2-user@YOUR_EC2_IP:/opt/shill-bidding-monitoring/
   ```

3. **Start the services**
   ```bash
   # SSH to the instance
   ssh -i ~/.ssh/shill-bidding-key ec2-user@YOUR_EC2_IP

   # Navigate to application directory
   cd /opt/shill-bidding-monitoring

   # Start all services
   docker-compose up -d
   ```

4. **Verify deployment**
   ```bash
   # Check service status
   docker-compose ps

   # Test API health
   curl http://YOUR_EC2_IP:8000/health

   # Test prediction endpoint
   curl -X POST "http://YOUR_EC2_IP:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "auction_id": "test_1",
       "bidder_tendency": 0.5,
       "bidding_ratio": 0.3,
       "successive_outbidding": 0.2,
       "last_bidding": 0.1,
       "auction_bids": 10,
       "starting_price_average": 100.0,
       "early_bidding": 0.4,
       "winning_ratio": 0.6,
       "auction_duration": 7
     }'
   ```

### Step 4: Monitoring Setup

1. **Access Grafana Dashboard**
   - URL: `http://YOUR_EC2_IP:3000`
   - Username: `admin`
   - Password: `admin`

2. **Access Prometheus**
   - URL: `http://YOUR_EC2_IP:9090`

3. **Generate traffic for monitoring**
   ```bash
   # SSH to the instance
   ssh -i ~/.ssh/shill-bidding-key ec2-user@YOUR_EC2_IP

   # Navigate to application directory
   cd /opt/shill-bidding-monitoring

   # Start traffic generation in background
   python3 monitoring/generate_traffic.py &
   ```

## 📁 Project Structure

```
shill-bidding-prediction/
├── .github/workflows/     # CI/CD pipeline
├── api/                   # FastAPI application
│   ├── main.py           # Main API application
│   ├── main_simple.py    # Simplified API for Docker
│   ├── Dockerfile        # Container configuration
│   └── requirements.txt  # Python dependencies
├── data/                  # Dataset files
│   ├── Shill Bidding Dataset.csv
│   └── Shill Bidding Dataset Excelfied.xlsx
├── docs/                  # Documentation and research
│   ├── AWS_DEPLOYMENT_SUMMARY.md
│   ├── aws-requirements.md
│   ├── README_MONITORING.md
│   └── AuctionFraudClassificationGAnSad.pdf
├── exploration/           # Jupyter notebooks
│   ├── eda.ipynb
│   ├── feature_eng.ipynb
│   └── register_model.ipynb
├── logs/                  # Log files
├── monitoring/            # Monitoring scripts and configs
│   ├── test_monitoring.py
│   ├── generate_traffic.py
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── pipeline/              # Data pipeline scripts
├── scripts/               # Utility scripts
│   ├── cleanup.sh
│   └── deploy.sh
├── src/                   # Core ML code
│   ├── models.py         # Model training classes
│   └── register_model_for_api.py
├── tests/                 # Test files
│   ├── unit/
│   └── integration/
├── terraform/             # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
├── docker-compose.yml     # Service orchestration
├── requirements.txt       # Python dependencies
├── Makefile              # Development commands
└── README.md             # This file
```

## 🧪 Testing

### Unit Tests
```bash
make test-unit
```

### Integration Tests
```bash
make test-integration
```

### All Tests with Coverage
```bash
make test-coverage
```

### Quick Tests
```bash
make quick-test
```

## 🔧 Development

### Code Formatting
```bash
make format
```

### Linting
```bash
make lint
```

### Security Scan
```bash
make security
```

### Full Development Workflow
```bash
make dev-test
```

## 🚀 Deployment Commands

### One-Command Deployment
```bash
# Deploy everything to AWS
make deploy-cloud

# Destroy when done
make destroy-cloud
```

### Manual Deployment Steps
```bash
# Train model
python src/register_model_for_api.py

# Deploy infrastructure
cd terraform && terraform apply -auto-approve

# Upload files and start services
./scripts/deploy.sh
```

## 🔍 Monitoring

### Health Check
```bash
curl http://YOUR_EC2_IP:8000/health
```

### Metrics
```bash
curl http://YOUR_EC2_IP:8000/metrics
```

### Performance Benchmark
```bash
make benchmark
```

## 📊 CI/CD Pipeline

The GitHub Actions workflow includes:

1. **Lint and Format Check**
   - Black code formatter
   - isort import sorter
   - Flake8 linter
   - MyPy type checker

2. **Unit Tests**
   - pytest with coverage
   - Codecov integration

3. **Integration Tests**
   - API endpoint testing
   - Performance testing
   - Error handling

4. **Security Scan**
   - Bandit security scanner
   - Vulnerability detection

5. **Terraform Plan**
   - Infrastructure validation
   - Change preview

6. **Deployment**
   - Staging (develop branch)
   - Production (main branch)

## 🛡️ Security

- **Code Security**: Bandit scanning for vulnerabilities
- **Infrastructure Security**: Terraform security best practices
- **API Security**: Input validation and error handling
- **Monitoring**: Security metrics and alerts

## 📈 Performance

- **API Response Time**: < 1 second
- **Concurrent Requests**: Tested with 5+ simultaneous requests
- **Resource Usage**: Optimized for t3.micro instance
- **Monitoring**: Real-time metrics and dashboards

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_PORT=8000
ENVIRONMENT=dev
PROJECT_NAME=shill-bidding

# AWS Configuration
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Terraform Variables
```hcl
# terraform/terraform.tfvars
aws_region = "eu-north-1"
instance_type = "t3.micro"
environment = "dev"
```

## 📚 API Documentation

### Endpoints

- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API docs

### Example Prediction Request
```json
{
  "auction_id": "test_auction",
  "bidder_tendency": 0.5,
  "bidding_ratio": 0.6,
  "successive_outbidding": 0.4,
  "last_bidding": 0.3,
  "auction_bids": 10,
  "starting_price_average": 100.0,
  "early_bidding": 0.2,
  "winning_ratio": 0.9,
  "auction_duration": 7
}
```

### Example Response
```json
{
  "prediction": 1,
  "probability": 0.85,
  "is_shill_bid": true,
  "drift_score": 0.92,
  "performance_score": 0.78
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Terraform Provider Version Conflict**
   ```bash
   cd terraform
   terraform init -upgrade
   ```

2. **S3 Bucket Not Found**
   ```bash
   aws s3 mb s3://shillbidding-s3 --region eu-north-1
   ```

3. **Tests Failing**
   ```bash
   make clean
   make install-dev
   make test
   ```

4. **Pre-commit Hooks Not Working**
   ```bash
   make install-hooks
   pre-commit run --all-files
   ```

5. **Docker Build Failing**
   ```bash
   # Check if files are in correct locations
   ls -la api/

   # Rebuild with no cache
   docker-compose build --no-cache
   ```

6. **SSH Connection Issues**
   ```bash
   # Check key permissions
   chmod 600 ~/.ssh/shill-bidding-key

   # Test connection
   ssh -i ~/.ssh/shill-bidding-key ec2-user@YOUR_EC2_IP
   ```

7. **Services Not Starting**
   ```bash
   # Check Docker logs
   docker-compose logs api
   docker-compose logs prometheus
   docker-compose logs grafana

   # Restart services
   docker-compose down && docker-compose up -d
   ```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Commit with conventional commits
7. Push and create a pull request

## 📊 Project Status

- ✅ Infrastructure deployed
- ✅ CI/CD pipeline active
- ✅ Monitoring operational
- ✅ Documentation complete

## 🌐 Live Demo

**Access the deployed application:**
- **API Endpoint**: `http://13.49.246.221:8000`
- **API Documentation**: `http://13.49.246.221:8000/docs`
- **Grafana Dashboard**: `http://13.49.246.221:3000` (admin/admin)
- **Prometheus**: `http://13.49.246.221:9090`
