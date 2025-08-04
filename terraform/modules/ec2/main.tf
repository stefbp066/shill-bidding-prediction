# EC2 Module for Application Server

resource "aws_instance" "app_server" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = var.security_group_ids
  key_name               = var.key_name
  iam_instance_profile   = var.iam_instance_profile

  # User data script
  user_data = var.user_data

  # Root volume
  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true

    tags = {
      Name = "${var.instance_name}-root-volume"
    }
  }

  # Monitoring
  monitoring = var.enable_monitoring

  tags = {
    Name = var.instance_name
  }

  lifecycle {
    create_before_destroy = true
  }
}

# EBS volume for additional storage (optional)
resource "aws_ebs_volume" "additional_storage" {
  count             = var.additional_volume_size > 0 ? 1 : 0
  availability_zone = aws_instance.app_server.availability_zone
  size              = var.additional_volume_size
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "${var.instance_name}-additional-volume"
  }
}

# Attach additional volume
resource "aws_volume_attachment" "additional_storage" {
  count       = var.additional_volume_size > 0 ? 1 : 0
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.additional_storage[0].id
  instance_id = aws_instance.app_server.id
}
