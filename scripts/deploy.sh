#!/bin/bash
# Deployment script for Shill Bidding Model Monitoring

set -e

echo "🚀 Starting deployment of Shill Bidding Model Monitoring..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
        exit 1
    fi

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}❌ Terraform not found. Please install it first.${NC}"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Train and register model
train_model() {
    echo -e "${BLUE}Training and registering model...${NC}"

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Run model training
    python src/register_model_for_api.py

    echo -e "${GREEN}✅ Model training completed${NC}"
}

# Deploy infrastructure
deploy_infrastructure() {
    echo -e "${BLUE}Deploying AWS infrastructure...${NC}"

    cd terraform

    # Initialize Terraform
    echo -e "${YELLOW}Initializing Terraform...${NC}"
    terraform init

    # Plan deployment
    echo -e "${YELLOW}Planning deployment...${NC}"
    terraform plan -out=tfplan

    # Apply deployment
    echo -e "${YELLOW}Applying deployment...${NC}"
    terraform apply tfplan

    # Get outputs
    echo -e "${YELLOW}Getting deployment outputs...${NC}"
    INSTANCE_IP=$(terraform output -raw instance_public_ip)
    INSTANCE_ID=$(terraform output -raw instance_id)

    echo -e "${GREEN}✅ Infrastructure deployed successfully${NC}"
    echo -e "${BLUE}Instance IP: ${INSTANCE_IP}${NC}"
    echo -e "${BLUE}Instance ID: ${INSTANCE_ID}${NC}"

    cd ..

    # Store outputs for later use
    echo "$INSTANCE_IP" > .deployment_ip
    echo "$INSTANCE_ID" > .deployment_id
}

# Wait for instance to be ready
wait_for_instance() {
    echo -e "${BLUE}Waiting for instance to be ready...${NC}"

    INSTANCE_IP=$(cat .deployment_ip)

    echo -e "${YELLOW}Waiting for SSH access...${NC}"
    until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP "echo 'SSH ready'" 2>/dev/null; do
        echo -e "${YELLOW}Waiting for SSH...${NC}"
        sleep 30
    done

    echo -e "${GREEN}✅ Instance is ready${NC}"
}

# Upload application files
upload_files() {
    echo -e "${BLUE}Uploading application files...${NC}"

    INSTANCE_IP=$(cat .deployment_ip)

    # Create remote directory
    ssh ec2-user@$INSTANCE_IP "mkdir -p /opt/shill-bidding-monitoring"

    # Upload files
    echo -e "${YELLOW}Uploading application files...${NC}"
    scp -r api ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp -r monitoring ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp -r data ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp -r mlruns ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp docker-compose.yml ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp api/Dockerfile ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/
    scp api/requirements.txt ec2-user@$INSTANCE_IP:/opt/shill-bidding-monitoring/

    echo -e "${GREEN}✅ Files uploaded successfully${NC}"
}

# Start the application
start_application() {
    echo -e "${BLUE}Starting the application...${NC}"

    INSTANCE_IP=$(cat .deployment_ip)

    # Start the monitoring stack
    ssh ec2-user@$INSTANCE_IP "cd /opt/shill-bidding-monitoring && docker-compose up -d"

    echo -e "${GREEN}✅ Application started successfully${NC}"
}

# Test the deployment
test_deployment() {
    echo -e "${BLUE}Testing the deployment...${NC}"

    INSTANCE_IP=$(cat .deployment_ip)

    # Wait for services to start
    echo -e "${YELLOW}Waiting for services to start...${NC}"
    sleep 60

    # Test API health
    echo -e "${YELLOW}Testing API health...${NC}"
    if curl -f http://$INSTANCE_IP:8000/health; then
        echo -e "${GREEN}✅ API is healthy${NC}"
    else
        echo -e "${RED}❌ API health check failed${NC}"
    fi

    # Test Grafana
    echo -e "${YELLOW}Testing Grafana...${NC}"
    if curl -f http://$INSTANCE_IP:3000; then
        echo -e "${GREEN}✅ Grafana is accessible${NC}"
    else
        echo -e "${RED}❌ Grafana check failed${NC}"
    fi

    # Test Prometheus
    echo -e "${YELLOW}Testing Prometheus...${NC}"
    if curl -f http://$INSTANCE_IP:9090; then
        echo -e "${GREEN}✅ Prometheus is accessible${NC}"
    else
        echo -e "${RED}❌ Prometheus check failed${NC}"
    fi
}

# Display deployment info
show_deployment_info() {
    echo -e "${BLUE}📊 Deployment Information${NC}"
    echo "=================================="

    INSTANCE_IP=$(cat .deployment_ip)

    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  🌐 API: http://$INSTANCE_IP:8000"
    echo -e "  📊 Grafana: http://$INSTANCE_IP:3000 (admin/admin123)"
    echo -e "  📈 Prometheus: http://$INSTANCE_IP:9090"
    echo ""
    echo -e "${BLUE}API Endpoints:${NC}"
    echo -e "  🔍 Health: http://$INSTANCE_IP:8000/health"
    echo -e "  📊 Metrics: http://$INSTANCE_IP:8000/metrics"
    echo -e "  🤖 Predict: http://$INSTANCE_IP:8000/predict"
    echo ""
    echo -e "${YELLOW}To test the API:${NC}"
    echo -e "  curl -X POST http://$INSTANCE_IP:8000/predict \\"
    echo -e "    -H 'Content-Type: application/json' \\"
    echo -e "    -d '{\"auction_id\":\"test\",\"bidder_tendency\":0.5,\"bidding_ratio\":0.5,\"successive_outbidding\":0.5,\"last_bidding\":0.5,\"auction_bids\":0.5,\"starting_price_average\":0.5,\"early_bidding\":0.5,\"winning_ratio\":0.5}'"
    echo ""
    echo -e "${YELLOW}To generate traffic:${NC}"
    echo -e "  python monitoring/generate_traffic.py"
    echo ""
    echo -e "${BLUE}To destroy the deployment:${NC}"
    echo -e "  cd terraform && terraform destroy"
}

# Main deployment function
main() {
    echo -e "${BLUE}🚀 Shill Bidding Model Monitoring Deployment${NC}"
    echo "=================================================="

    check_prerequisites
    train_model
    deploy_infrastructure
    wait_for_instance
    upload_files
    start_application
    test_deployment
    show_deployment_info
}

# Run main function
main "$@"
