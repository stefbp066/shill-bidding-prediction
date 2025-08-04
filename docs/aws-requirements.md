# AWS Deployment Requirements for Shill Bidding Model Monitoring

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EC2 Instance  │───▶│   Prometheus    │───▶│    Grafana      │
│   (API + Model) │    │   (EC2/ECS)     │    │   (EC2/ECS)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Bucket     │    │   CloudWatch    │    │   Application   │
│   (Model Files) │    │   (Logs/Metrics)│    │   Load Balancer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 **AWS Services Required**

### **Core Infrastructure**
- **EC2**: Application servers for API, Prometheus, Grafana
- **S3**: Model storage, monitoring data, logs
- **VPC**: Network isolation and security
- **Security Groups**: Firewall rules
- **IAM**: Permissions and roles

### **Optional/Advanced**
- **ECS/Fargate**: Container orchestration (alternative to EC2)
- **RDS**: Database for Grafana (alternative to SQLite)
- **ElastiCache**: Redis for session management
- **CloudWatch**: Centralized logging and monitoring
- **Application Load Balancer**: Traffic distribution
- **Route 53**: DNS management

## 🎯 **Minimum Viable Deployment**

### **1. EC2 Instances**
```
┌─────────────────────────────────────────────────────────────┐
│                    EC2 Instance                            │
├─────────────────────────────────────────────────────────────┤
│ • API Service (FastAPI + Model)                           │
│ • Prometheus (Metrics Collection)                          │
│ • Grafana (Visualization)                                 │
│ • Docker & Docker Compose                                 │
└─────────────────────────────────────────────────────────────┘
```

### **2. S3 Bucket**
```
┌─────────────────────────────────────────────────────────────┐
│                    S3 Bucket                              │
├─────────────────────────────────────────────────────────────┤
│ • /models/ - Trained model files                          │
│ • /monitoring/ - Evidently reports                        │
│ • /logs/ - Application logs                               │
│ • /data/ - Reference datasets                             │
└─────────────────────────────────────────────────────────────┘
```

### **3. VPC & Security**
```
┌─────────────────────────────────────────────────────────────┐
│                    VPC                                    │
├─────────────────────────────────────────────────────────────┤
│ • Public Subnet (EC2)                                     │
│ • Security Group (Ports: 22, 80, 443, 8000, 9090, 3000)  │
│ • Internet Gateway                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 **Terraform Structure**

```
terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Input variables
├── outputs.tf             # Output values
├── providers.tf           # AWS provider configuration
├── modules/
│   ├── vpc/              # VPC and networking
│   ├── ec2/              # EC2 instance setup
│   ├── s3/               # S3 bucket configuration
│   ├── security/         # Security groups and IAM
│   └── monitoring/       # Prometheus/Grafana setup
└── scripts/
    ├── user-data.sh      # EC2 initialization script
    └── docker-compose.yml # Container orchestration
```

## 📦 **Deployment Components**

### **1. EC2 Instance Requirements**
- **Instance Type**: t3.medium or t3.large (2-4 vCPU, 4-8 GB RAM)
- **OS**: Amazon Linux 2 or Ubuntu 20.04
- **Storage**: 20-50 GB EBS volume
- **Docker**: Pre-installed with Docker Compose

### **2. S3 Bucket Structure**
```
s3://your-model-monitoring-bucket/
├── models/
│   ├── shill-bidding-model/
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── evidently-reports/
├── monitoring/
│   ├── prometheus-data/
│   ├── grafana-dashboards/
│   └── evidently-reports/
└── logs/
    ├── api/
    ├── prometheus/
    └── grafana/
```

### **3. Security Requirements**
- **IAM Role**: EC2 instance profile for S3 access
- **Security Group**: Allow HTTP/HTTPS and monitoring ports
- **SSL Certificate**: For HTTPS (optional but recommended)

## 🚀 **Deployment Steps**

### **Phase 1: Infrastructure**
1. Create VPC with public subnet
2. Create S3 bucket for model storage
3. Create IAM roles and security groups
4. Launch EC2 instance with user data script

### **Phase 2: Application Deployment**
1. Install Docker and Docker Compose
2. Pull application code from Git
3. Configure environment variables
4. Start monitoring stack

### **Phase 3: Monitoring Setup**
1. Configure Prometheus data source
2. Import Grafana dashboards
3. Set up S3 integration for data persistence
4. Configure alerts and notifications

## 💰 **Cost Estimation**

### **Monthly Costs (US East)**
- **EC2 t3.medium**: ~$30/month
- **S3 Storage (100GB)**: ~$2.30/month
- **Data Transfer**: ~$5-10/month
- **Total**: ~$40-50/month

### **Scaling Options**
- **EC2 t3.large**: ~$60/month (more resources)
- **ECS Fargate**: ~$80-120/month (managed containers)
- **RDS**: ~$30-50/month (managed database)

## 🔒 **Security Considerations**

### **Network Security**
- VPC with public/private subnets
- Security groups limiting access
- HTTPS with SSL certificates
- VPN access for admin functions

### **Data Security**
- S3 bucket encryption
- IAM roles with least privilege
- Secrets management (AWS Secrets Manager)
- CloudTrail for audit logging

### **Application Security**
- Docker image scanning
- Regular security updates
- Monitoring for suspicious activity
- Backup and disaster recovery

## 📊 **Monitoring & Alerting**

### **AWS CloudWatch Integration**
- Application metrics
- System metrics (CPU, memory, disk)
- Custom business metrics
- Log aggregation and analysis

### **Grafana Dashboards**
- Model performance metrics
- Data drift detection
- System health monitoring
- Business KPIs

### **Alerting**
- High drift scores
- Model performance degradation
- System resource issues
- Error rate spikes

## 🛠️ **Operational Considerations**

### **Backup Strategy**
- S3 versioning for model files
- EBS snapshots for instance data
- Cross-region replication for critical data
- Automated backup schedules

### **Scaling Strategy**
- Auto Scaling Groups for EC2
- Load balancer for multiple instances
- Container orchestration with ECS
- Microservices architecture

### **Disaster Recovery**
- Multi-AZ deployment
- Automated failover procedures
- Data replication across regions
- Recovery time objectives (RTO/RPO)

## 📝 **Next Steps**

1. **Create Terraform Configuration**
2. **Set up AWS CLI and credentials**
3. **Test deployment in staging environment**
4. **Configure monitoring and alerting**
5. **Document operational procedures**
6. **Plan production deployment**

This setup provides a solid foundation for running the monitoring stack in AWS while maintaining flexibility for future scaling and enhancements.
