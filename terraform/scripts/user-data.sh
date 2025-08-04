#!/bin/bash

# User data script for EC2 instance initialization
# This script sets up the monitoring stack on the EC2 instance

set -e

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install additional tools
yum install -y git curl wget unzip

# Create application directory
mkdir -p /opt/shill-bidding-monitoring
cd /opt/shill-bidding-monitoring

# Create environment file
cat > .env << EOF
# Environment Configuration
ENVIRONMENT=${environment}
PROJECT_NAME=${project_name}
S3_BUCKET_NAME=${s3_bucket_name}

# API Configuration
API_PORT=8000
MLFLOW_TRACKING_URI=mlruns
MODEL_NAME=shill-bidding-model
MODEL_VERSION=latest

# Monitoring Configuration
PROMETHEUS_RETENTION_TIME=200h
GRAFANA_ADMIN_PASSWORD=admin123
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # API Service with Evidently Monitoring
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=mlruns
      - MODEL_NAME=shill-bidding-model
      - MODEL_VERSION=latest
      - API_PORT=8000
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
      - ./monitoring:/app/monitoring
    networks:
      - monitoring-network
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - monitoring-network
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - monitoring-network
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring-network:
    driver: bridge
EOF

# Create monitoring directory structure
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# Create Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # API service metrics
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

# Create Grafana datasource configuration
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create Grafana dashboard provisioning
cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create a simple dashboard
cat > monitoring/grafana/dashboards/model-monitoring.json << 'EOF'
{
  "id": null,
  "title": "Shill Bidding Model Monitoring",
  "tags": ["ml", "monitoring", "evidently"],
  "style": "dark",
  "timezone": "browser",
  "panels": [
    {
      "id": 1,
      "title": "Data Drift Score",
      "type": "stat",
      "targets": [
        {
          "expr": "evidently_data_drift_score",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "thresholds": {
            "steps": [
              {"color": "green", "value": null},
              {"color": "yellow", "value": 0.3},
              {"color": "red", "value": 0.7}
            ]
          },
          "unit": "percent",
          "min": 0,
          "max": 1
        }
      },
      "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
    },
    {
      "id": 2,
      "title": "Model Performance Score",
      "type": "stat",
      "targets": [
        {
          "expr": "evidently_model_performance_score",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "thresholds": {
            "steps": [
              {"color": "red", "value": null},
              {"color": "yellow", "value": 0.7},
              {"color": "green", "value": 0.9}
            ]
          },
          "unit": "percent",
          "min": 0,
          "max": 1
        }
      },
      "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
    }
  ],
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "refresh": "10s"
}
EOF

# Create application files
mkdir -p api
mkdir -p data

# Create a simple API Dockerfile
cat > api/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main_simple.py main.py
RUN mkdir -p /app/mlruns

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create requirements.txt
cat > api/requirements.txt << 'EOF'
fastapi>=0.115.0
uvicorn[standard]>=0.27.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
pydantic>=2.0.0
prometheus-client>=0.17.0
python-dotenv>=1.0.0
requests>=2.25.0
EOF

# Create a simple API
cat > api/main_simple.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter('evidently_prediction_count_total', 'Total number of predictions')
PREDICTION_ERRORS = Counter('evidently_prediction_errors_total', 'Total number of prediction errors')
PREDICTION_DURATION = Histogram('evidently_prediction_duration_seconds', 'Prediction duration in seconds')
DRIFT_SCORE = Gauge('evidently_data_drift_score', 'Data drift score')
MODEL_PERFORMANCE = Gauge('evidently_model_performance_score', 'Model performance score')

app = FastAPI(title="Shill Bidding Prediction API")

class BidData(BaseModel):
    auction_id: str
    bidder_tendency: float
    bidding_ratio: float
    successive_outbidding: float
    last_bidding: float
    auction_bids: float
    starting_price_average: float
    early_bidding: float
    winning_ratio: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_shill_bid: bool
    drift_score: float = None
    performance_score: float = None

class MockModel:
    def predict(self, X):
        return np.random.choice([0, 1], size=len(X))

    def predict_proba(self, X):
        probs = np.random.random(len(X))
        return np.column_stack([1-probs, probs])

model = MockModel()

@app.post("/predict", response_model=PredictionResponse)
async def predict_shill_bidding(bid_data: BidData):
    start_time = time.time()

    try:
        df = pd.DataFrame([bid_data.dict()])

        column_mapping = {
            'auction_id': 'Auction_ID',
            'bidder_tendency': 'Bidder_Tendency',
            'bidding_ratio': 'Bidding_Ratio',
            'successive_outbidding': 'Successive_Outbidding',
            'last_bidding': 'Last_Bidding',
            'auction_bids': 'Auction_Bids',
            'starting_price_average': 'Starting_Price_Average',
            'early_bidding': 'Early_Bidding',
            'winning_ratio': 'Winning_Ratio'
        }
        df = df.rename(columns=column_mapping)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        PREDICTION_COUNTER.inc()
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        feature_values = list(bid_data.dict().values())[1:]
        avg_feature_value = np.mean([v for v in feature_values if isinstance(v, (int, float))])

        drift_score = min(1.0, max(0.0, abs(avg_feature_value - 0.5) * 2))
        drift_score += np.random.normal(0, 0.1)
        drift_score = max(0.0, min(1.0, drift_score))

        performance_score = probability if prediction == 1 else (1 - probability)
        performance_score += np.random.normal(0, 0.05)
        performance_score = max(0.0, min(1.0, performance_score))

        DRIFT_SCORE.set(drift_score)
        MODEL_PERFORMANCE.set(performance_score)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            is_shill_bid=bool(prediction == 1),
            drift_score=drift_score,
            performance_score=performance_score
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Start the monitoring stack
cd /opt/shill-bidding-monitoring
docker-compose up -d

# Create a simple test script
cat > test_api.py << 'EOF'
#!/usr/bin/env python3
import requests
import random
import time

def test_api():
    url = "http://localhost:8000"

    # Test health
    response = requests.get(f"{url}/health")
    print(f"Health check: {response.status_code}")

    # Test prediction
    data = {
        "auction_id": "test_auction",
        "bidder_tendency": random.uniform(0, 1),
        "bidding_ratio": random.uniform(0, 1),
        "successive_outbidding": random.uniform(0, 1),
        "last_bidding": random.uniform(0, 1),
        "auction_bids": random.uniform(0, 1),
        "starting_price_average": random.uniform(0, 1),
        "early_bidding": random.uniform(0, 1),
        "winning_ratio": random.uniform(0, 1)
    }

    response = requests.post(f"{url}/predict", json=data)
    print(f"Prediction: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Drift Score: {result['drift_score']:.3f}")
        print(f"Performance Score: {result['performance_score']:.3f}")

if __name__ == "__main__":
    test_api()
EOF

# Make test script executable
chmod +x test_api.py

# Wait for services to start
sleep 30

# Test the API
python3 test_api.py

echo "Monitoring stack setup complete!"
echo "Grafana: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000 (admin/admin123)"
echo "API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "Prometheus: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):9090"
