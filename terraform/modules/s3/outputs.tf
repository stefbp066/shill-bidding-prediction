# Outputs for S3 Module

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = data.aws_s3_bucket.existing_bucket.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = data.aws_s3_bucket.existing_bucket.arn
}

output "bucket_region" {
  description = "Region of the S3 bucket"
  value       = data.aws_s3_bucket.existing_bucket.region
}
