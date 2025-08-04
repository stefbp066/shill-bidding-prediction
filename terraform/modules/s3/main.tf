# S3 Module for Model Storage and Monitoring Data

# Use existing bucket
data "aws_s3_bucket" "existing_bucket" {
  bucket = var.bucket_name
}

# Bucket versioning (only if not already configured)
resource "aws_s3_bucket_versioning" "model_storage" {
  bucket = data.aws_s3_bucket.existing_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Bucket encryption (only if not already configured)
resource "aws_s3_bucket_server_side_encryption_configuration" "model_storage" {
  bucket = data.aws_s3_bucket.existing_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Bucket public access block (only if not already configured)
resource "aws_s3_bucket_public_access_block" "model_storage" {
  bucket = data.aws_s3_bucket.existing_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket lifecycle configuration (only if not already configured)
resource "aws_s3_bucket_lifecycle_configuration" "model_storage" {
  bucket = data.aws_s3_bucket.existing_bucket.id

  rule {
    id     = "model-retention"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Create folder structure
resource "aws_s3_object" "models_folder" {
  bucket = data.aws_s3_bucket.existing_bucket.id
  key    = "models/"
  source = "/dev/null"
}

resource "aws_s3_object" "monitoring_folder" {
  bucket = data.aws_s3_bucket.existing_bucket.id
  key    = "monitoring/"
  source = "/dev/null"
}

resource "aws_s3_object" "logs_folder" {
  bucket = data.aws_s3_bucket.existing_bucket.id
  key    = "logs/"
  source = "/dev/null"
}

resource "aws_s3_object" "data_folder" {
  bucket = data.aws_s3_bucket.existing_bucket.id
  key    = "data/"
  source = "/dev/null"
}
