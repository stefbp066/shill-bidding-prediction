# Main Terraform configuration for Shill Bidding Model Monitoring Stack
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "shill-bidding-monitoring"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI (Amazon Linux 2) *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"

  vpc_name           = "${var.project_name}-vpc"
  vpc_cidr           = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
  public_subnets     = var.public_subnets
  environment        = var.environment
}

# Use existing S3 bucket
data "aws_s3_bucket" "existing_bucket" {
  bucket = "shillbidding-s3"
}

# S3 Bucket for Model Storage (using existing bucket)
module "s3" {
  source = "./modules/s3"

  bucket_name = data.aws_s3_bucket.existing_bucket.bucket
  environment = var.environment
}

# Security Groups and IAM
module "security" {
  source = "./modules/security"

  vpc_id      = module.vpc.vpc_id
  environment = var.environment
  project_name = var.project_name
}

# EC2 Instance for Application
module "ec2" {
  source = "./modules/ec2"

  instance_name     = "${var.project_name}-monitoring"
  instance_type     = var.instance_type
  ami_id            = data.aws_ami.deep_learning.id
  subnet_id         = module.vpc.public_subnet_ids[0]
  security_group_ids = [module.security.app_security_group_id]
  key_name          = var.key_pair_name

  # Volume size for Deep Learning AMI (minimum 105GB)
  volume_size = var.instance_volume_size

  # IAM role for S3 access
  iam_instance_profile = module.security.ec2_instance_profile_name

  # User data script
  user_data = templatefile("${path.module}/scripts/user-data.sh", {
    s3_bucket_name = module.s3.bucket_name
    environment    = var.environment
    project_name   = var.project_name
  })

  environment = var.environment
  project_name = var.project_name
}

# Random string for unique bucket names (not needed for existing bucket)
# resource "random_string" "bucket_suffix" {
#   length  = 8
#   special = false
#   upper   = false
# }

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/aws/ec2/${var.project_name}-${var.environment}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-${var.environment}-logs"
  }
}

# Outputs
output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = module.ec2.public_ip
}

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = module.ec2.instance_id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for model storage"
  value       = module.s3.bucket_name
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "grafana_url" {
  description = "URL for Grafana dashboard"
  value       = "http://${module.ec2.public_ip}:3000"
}

output "api_url" {
  description = "URL for the API"
  value       = "http://${module.ec2.public_ip}:8000"
}

output "prometheus_url" {
  description = "URL for Prometheus"
  value       = "http://${module.ec2.public_ip}:9090"
}
