# AWS Deployment Guide for Shill Bidding Model Monitoring

## 🚀 **Quick Start**

### **Prerequisites**
- AWS CLI configured with appropriate permissions
- Terraform installed (version >= 1.0)
- SSH key pair for EC2 access

### **1. Configure AWS Credentials**
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
```

### **2. Create SSH Key Pair (if needed)**
```bash
aws ec2 create-key-pair --key-name shill-bidding-key --query 'KeyMaterial' --output text > shill-bidding-key.pem
chmod 400 shill-bidding-key.pem
```

### **3. Deploy Infrastructure**
```bash
cd terraform

# Initialize Terraform
terraform init

# Plan the deployment
terraform plan

# Apply the configuration
terraform apply
```

### **4. Access Your Monitoring Stack**
After deployment completes, you'll see output like:
```
grafana_url = "http://54.123.45.67:3000"
api_url = "http://54.123.45.67:8000"
prometheus_url = "http://54.123.45.67:9090"
```

## 📋 **What Gets Deployed**

### **Infrastructure Components**
- **VPC**: Custom VPC with public subnets
- **EC2 Instance**: t3.medium with Docker and monitoring stack
- **S3 Bucket**: For model storage and monitoring data
- **Security Groups**: Firewall rules for web access
- **IAM Role**: EC2 instance profile for S3 access

### **Application Components**
- **API Service**: FastAPI with Prometheus metrics
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard visualization
- **Docker Compose**: Container orchestration

## 🔧 **Configuration Options**

### **Environment Variables**
Create a `terraform.tfvars` file to customize deployment:

```hcl
# Basic Configuration
aws_region = "us-east-1"
project_name = "shill-bidding"
environment = "dev"

# Instance Configuration
instance_type = "t3.medium"
key_pair_name = "shill-bidding-key"

# Network Configuration
vpc_cidr = "10.0.0.0/16"
public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

# Security Configuration
allowed_ssh_cidr = ["YOUR_IP/32"]  # Restrict SSH access
allowed_web_cidr = ["0.0.0.0/0"]   # Allow web access from anywhere
```

### **Cost Optimization**
For development/testing, use smaller instances:
```hcl
instance_type = "t3.small"  # ~$15/month instead of ~$30/month
```

## 🔒 **Security Considerations**

### **Network Security**
- VPC with public subnets only (for simplicity)
- Security groups limiting access to required ports
- Consider private subnets for production

### **Access Control**
- SSH access restricted to your IP (recommended)
- Web access open for demo purposes
- IAM roles with least privilege

### **Data Security**
- S3 bucket with encryption enabled
- No public access to S3 bucket
- Versioning enabled for data protection

## 📊 **Monitoring & Access**

### **Grafana Dashboard**
- **URL**: `http://<EC2_IP>:3000`
- **Username**: `admin`
- **Password**: `admin123`
- **Features**: Model monitoring, drift detection, performance metrics

### **API Endpoints**
- **Health Check**: `http://<EC2_IP>:8000/health`
- **Predictions**: `http://<EC2_IP>:8000/predict`
- **Metrics**: `http://<EC2_IP>:8000/metrics`
- **Documentation**: `http://<EC2_IP>:8000/docs`

### **Prometheus**
- **URL**: `http://<EC2_IP>:9090`
- **Features**: Metrics collection, query interface, alerting

## 🛠️ **Operations**

### **SSH Access**
```bash
ssh -i shill-bidding-key.pem ec2-user@<EC2_IP>
```

### **View Logs**
```bash
# SSH into instance
ssh -i shill-bidding-key.pem ec2-user@<EC2_IP>

# Check Docker containers
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

### **Restart Services**
```bash
# SSH into instance
cd /opt/shill-bidding-monitoring
docker-compose restart
```

### **Update Application**
```bash
# SSH into instance
cd /opt/shill-bidding-monitoring
git pull  # if using git
docker-compose up -d --build
```

## 💰 **Cost Management**

### **Monthly Costs (US East)**
- **EC2 t3.medium**: ~$30/month
- **S3 Storage (100GB)**: ~$2.30/month
- **Data Transfer**: ~$5-10/month
- **Total**: ~$40-50/month

### **Cost Optimization Tips**
1. **Use Spot Instances**: Save 60-90% on EC2 costs
2. **Schedule Shutdown**: Stop instances when not in use
3. **Use Smaller Instances**: t3.small for development
4. **Monitor Usage**: Set up CloudWatch billing alerts

## 🧹 **Cleanup**

### **Destroy Infrastructure**
```bash
cd terraform
terraform destroy
```

### **Manual Cleanup**
```bash
# Delete S3 bucket contents
aws s3 rm s3://<bucket-name> --recursive

# Terminate EC2 instance
aws ec2 terminate-instances --instance-ids <instance-id>

# Delete key pair
aws ec2 delete-key-pair --key-name shill-bidding-key
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Instance Not Starting**
   ```bash
   # Check instance status
   aws ec2 describe-instances --instance-ids <instance-id>

   # Check user data logs
   ssh -i key.pem ec2-user@<IP>
   sudo cat /var/log/cloud-init-output.log
   ```

2. **Services Not Accessible**
   ```bash
   # Check security groups
   aws ec2 describe-security-groups --group-ids <sg-id>

   # Check if ports are open
   telnet <EC2_IP> 3000  # Grafana
   telnet <EC2_IP> 8000  # API
   telnet <EC2_IP> 9090  # Prometheus
   ```

3. **Docker Issues**
   ```bash
   # SSH into instance
   sudo systemctl status docker
   docker ps
   docker-compose logs
   ```

### **Useful Commands**
```bash
# Get instance public IP
terraform output instance_public_ip

# Get S3 bucket name
terraform output s3_bucket_name

# View all outputs
terraform output
```

## 📈 **Scaling Options**

### **Horizontal Scaling**
- Use Auto Scaling Groups
- Add Application Load Balancer
- Deploy multiple EC2 instances

### **Vertical Scaling**
- Upgrade to t3.large or t3.xlarge
- Add more EBS storage
- Use RDS for Grafana database

### **Container Orchestration**
- Migrate to ECS/Fargate
- Use Kubernetes (EKS)
- Implement service mesh

## 🎯 **Next Steps**

1. **Test the Deployment**: Verify all services are working
2. **Configure Alerts**: Set up CloudWatch alarms
3. **Add SSL**: Configure HTTPS with ACM certificate
4. **Implement CI/CD**: Automate deployments
5. **Add Monitoring**: Set up CloudWatch dashboards
6. **Security Hardening**: Implement private subnets, VPN access

This deployment provides a solid foundation for running your monitoring stack in AWS with room for future enhancements and scaling.
