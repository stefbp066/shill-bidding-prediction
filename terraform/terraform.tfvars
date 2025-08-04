# Terraform variables for Shill Bidding Model Monitoring
# Customize these values as needed

# AWS Configuration
aws_region = "eu-north-1"  # Changed to match your S3 bucket region

# Project Configuration
project_name = "shill-bidding"
environment  = "dev"

# Instance Configuration
instance_type = "t3.micro"  # Free tier eligible
key_pair_name = ""  # Leave empty if you don't have a key pair yet

# Network Configuration
vpc_cidr = "10.0.0.0/16"
public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]

# Security Configuration
allowed_ssh_cidr = ["0.0.0.0/0"]  # Allow SSH from anywhere (change to your IP for security)
allowed_web_cidr = ["0.0.0.0/0"]  # Allow web access from anywhere

# Storage Configuration
instance_volume_size = 120  # Increased for Deep Learning AMI (minimum 105GB)

# Monitoring Configuration
enable_monitoring = true
enable_backup = false  # Disabled to save costs
backup_retention_days = 7

# Additional tags
tags = {
  Owner       = "Your Name"
  Purpose     = "Shill Bidding Model Monitoring"
  CostCenter  = "ML-Research"
}
