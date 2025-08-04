# 🚀 Cloud Deployment Guide

## **Quick Start: One-Command Deployment**

```bash
# Deploy everything to AWS
make deploy-cloud

# Destroy when done
make destroy-cloud
```

## **📋 Prerequisites**

### **1. AWS Setup**
```bash
# Install AWS CLI
brew install awscli  # macOS
# or
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

### **2. Terraform Setup**
```bash
# Install Terraform
brew install terraform  # macOS
# or
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform
```

### **3. SSH Key Setup**
```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Add to AWS (replace with your key name)
aws ec2 import-key-pair --key-name "shill-bidding-key" --public-key-material fileb://~/.ssh/id_rsa.pub
```

## **🎯 Deployment Options**

### **Option 1: Automated Deployment (Recommended)**
```bash
# One command does everything
make deploy-cloud
```

**What this does:**
1. ✅ Trains and registers the ML model
2. ✅ Deploys AWS infrastructure (EC2, VPC, S3, Security Groups)
3. ✅ Uploads application files to EC2
4. ✅ Starts Docker Compose stack
5. ✅ Tests all services
6. ✅ Shows access URLs

### **Option 2: Manual Step-by-Step**
```bash
# 1. Train model
python src/register_model_for_api.py

# 2. Deploy infrastructure
cd terraform
terraform init
terraform plan
terraform apply

# 3. Get instance IP
terraform output instance_public_ip

# 4. Upload files (replace IP)
scp -r api ec2-user@YOUR_IP:/opt/shill-bidding-monitoring/
scp -r monitoring ec2-user@YOUR_IP:/opt/shill-bidding-monitoring/
scp -r data ec2-user@YOUR_IP:/opt/shill-bidding-monitoring/
scp -r mlruns ec2-user@YOUR_IP:/opt/shill-bidding-monitoring/
scp docker-compose.yml ec2-user@YOUR_IP:/opt/shill-bidding-monitoring/

# 5. Start services
ssh ec2-user@YOUR_IP "cd /opt/shill-bidding-monitoring && docker-compose up -d"
```

## **🌐 Access Your Deployment**

After deployment, you'll have access to:

### **API Endpoints**
- **Health Check**: `http://YOUR_IP:8000/health`
- **API Documentation**: `http://YOUR_IP:8000/docs`
- **Predictions**: `http://YOUR_IP:8000/predict`
- **Metrics**: `http://YOUR_IP:8000/metrics`

### **Monitoring Dashboards**
- **Grafana**: `http://YOUR_IP:3000` (admin/admin123)
- **Prometheus**: `http://YOUR_IP:9090`

### **Test the API**
```bash
# Test prediction
curl -X POST http://YOUR_IP:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "auction_id": "test_auction",
    "bidder_tendency": 0.5,
    "bidding_ratio": 0.5,
    "successive_outbidding": 0.5,
    "last_bidding": 0.5,
    "auction_bids": 0.5,
    "starting_price_average": 0.5,
    "early_bidding": 0.5,
    "winning_ratio": 0.5
  }'
```

## **📊 Monitoring Features**

### **Real-time Metrics**
- **Data Drift Score**: 0.0-1.0 (higher = more drift)
- **Model Performance**: 0.0-1.0 (higher = better performance)
- **Prediction Count**: Total predictions made
- **Error Rate**: Failed predictions percentage
- **Response Time**: API latency metrics

### **Grafana Dashboards**
- **Model Monitoring**: Real-time drift and performance
- **API Metrics**: Request volume and errors
- **System Health**: CPU, memory, disk usage

## **🔧 Configuration**

### **Environment Variables**
```bash
# API Configuration
API_PORT=8000
MLFLOW_TRACKING_URI=mlruns
MODEL_NAME=shill-bidding-model
MODEL_VERSION=latest

# Monitoring Configuration
PROMETHEUS_RETENTION_TIME=200h
GRAFANA_ADMIN_PASSWORD=admin123
```

### **Instance Types**
- **Development**: `t3.medium` (~$30/month)
- **Production**: `t3.large` or `c5.large` (~$60-80/month)
- **High Performance**: `c5.xlarge` (~$160/month)

## **💰 Cost Estimation**

### **Monthly Costs (t3.medium)**
- **EC2 Instance**: ~$30/month
- **S3 Storage**: ~$2.30/month (100GB)
- **Data Transfer**: ~$5-10/month
- **Total**: ~$40-50/month

### **Cost Optimization**
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID

# Start when needed
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID
```

## **🛠️ Troubleshooting**

### **Common Issues**

#### **1. SSH Connection Failed**
```bash
# Check security group allows SSH
aws ec2 describe-security-groups --group-names shill-bidding-monitoring

# Check instance status
aws ec2 describe-instances --instance-ids YOUR_INSTANCE_ID
```

#### **2. Docker Services Not Starting**
```bash
# SSH into instance
ssh ec2-user@YOUR_IP

# Check Docker status
sudo systemctl status docker

# Check container logs
docker-compose logs api
docker-compose logs prometheus
docker-compose logs grafana
```

#### **3. API Not Responding**
```bash
# Check if containers are running
docker ps

# Check API logs
docker-compose logs api

# Test API directly
curl http://localhost:8000/health
```

### **Useful Commands**
```bash
# View all logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# Check disk space
df -h

# Check memory usage
free -h
```

## **🧹 Cleanup**

### **Destroy Everything**
```bash
# One command cleanup
make destroy-cloud

# Or manual cleanup
cd terraform
terraform destroy -auto-approve
```

### **Partial Cleanup**
```bash
# Stop services only
ssh ec2-user@YOUR_IP "cd /opt/shill-bidding-monitoring && docker-compose down"

# Remove containers and volumes
ssh ec2-user@YOUR_IP "cd /opt/shill-bidding-monitoring && docker-compose down -v"
```

## **🚀 Next Steps**

### **Production Considerations**
1. **SSL/TLS**: Add HTTPS with Let's Encrypt
2. **Load Balancer**: Use ALB for high availability
3. **Auto Scaling**: Set up auto-scaling groups
4. **Backup**: Configure automated backups
5. **Monitoring**: Set up CloudWatch alarms

### **Advanced Features**
1. **CI/CD**: Connect to GitHub Actions
2. **Blue/Green**: Implement zero-downtime deployments
3. **Multi-region**: Deploy to multiple regions
4. **CDN**: Add CloudFront for global distribution

---

**🎉 You're ready to deploy! Run `make deploy-cloud` to get started.**
