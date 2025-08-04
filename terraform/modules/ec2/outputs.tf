# Outputs for EC2 Module

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.app_server.id
}

output "public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}

output "private_ip" {
  description = "Private IP address of the EC2 instance"
  value       = aws_instance.app_server.private_ip
}

output "availability_zone" {
  description = "Availability zone of the EC2 instance"
  value       = aws_instance.app_server.availability_zone
}

output "instance_arn" {
  description = "ARN of the EC2 instance"
  value       = aws_instance.app_server.arn
}
