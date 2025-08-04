#!/bin/bash

# Monitoring Stack Startup Script
# This script starts the complete monitoring stack for the Shill Bidding Prediction model

echo "🚀 Starting Shill Bidding Model Monitoring Stack"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

echo "📋 Checking prerequisites..."
echo "✅ Docker is running"
echo "✅ docker-compose is available"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file with default values..."
    cat > .env << EOF
# Evidently Cloud (optional - set your token if you have one)
# EVIDENTLY_TOKEN=your_evidently_cloud_token

# API Configuration
API_PORT=8000
MLFLOW_TRACKING_URI=mlruns
MODEL_NAME=shill-bidding-model
MODEL_VERSION=latest
EOF
    echo "✅ Created .env file"
fi

echo ""
echo "🔧 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Service URLs:"
echo "   API:          http://localhost:8000"
echo "   Grafana:      http://localhost:3000 (admin/admin)"
echo "   Prometheus:   http://localhost:9090"

echo ""
echo "📖 Quick Start Guide:"
echo "   1. Open Grafana at http://localhost:3000"
echo "   2. Login with admin/admin"
echo "   3. The Model Monitoring dashboard should be pre-loaded"
echo "   4. Test the API with: python test_monitoring.py"

echo ""
echo "🔍 Useful Commands:"
echo "   View logs:        docker-compose logs -f"
echo "   Stop services:    docker-compose down"
echo "   Restart:          docker-compose restart"
echo "   Test monitoring:  python test_monitoring.py"

echo ""
echo "✅ Monitoring stack is ready!"
echo "================================================"
