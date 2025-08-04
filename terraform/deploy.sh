#!/bin/bash

# Shill Bidding Model Monitoring - Deployment Script
# This script deploys the monitoring infrastructure to AWS

set -e  # Exit on any error

echo "🚀 Starting Shill Bidding Model Monitoring Deployment..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed. Please install Terraform first."
    exit 1
fi

# Check if the S3 bucket exists
echo "🔍 Checking if S3 bucket 'shillbidding-s3' exists..."
if ! aws s3 ls s3://shillbidding-s3 &> /dev/null; then
    echo "❌ S3 bucket 'shillbidding-s3' does not exist. Please create it first:"
    echo "   aws s3 mb s3://shillbidding-s3"
    exit 1
fi

echo "✅ S3 bucket 'shillbidding-s3' exists"

# Initialize Terraform
echo "🔧 Initializing Terraform..."
terraform init

# Plan the deployment
echo "📋 Planning deployment..."
terraform plan

# Ask for confirmation
echo ""
read -p "🤔 Do you want to proceed with the deployment? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled."
    exit 1
fi

# Apply the configuration
echo "🚀 Deploying infrastructure..."
terraform apply -auto-approve

# Show outputs
echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
terraform output

echo ""
echo "🔗 Quick Access:"
echo "   - Grafana Dashboard: $(terraform output -raw grafana_url)"
echo "   - API Documentation: $(terraform output -raw api_url)/docs"
echo "   - Prometheus: $(terraform output -raw prometheus_url)"
echo ""
echo "📝 Next Steps:"
echo "   1. Wait 5-10 minutes for the instance to fully initialize"
echo "   2. Access Grafana at: $(terraform output -raw grafana_url)"
echo "   3. Default credentials: admin/admin123"
echo "   4. Test the API at: $(terraform output -raw api_url)/health"
echo ""
echo "💡 To destroy the infrastructure when done:"
echo "   terraform destroy"
