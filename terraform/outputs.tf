output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = data.aws_s3_bucket.existing_bucket.id
}
