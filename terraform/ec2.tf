# Tự động tìm AMI Ubuntu 22.04 LTS mới nhất (Canonical account ID)
data "aws_ami" "ubuntu_22" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "backend" {
  ami                    = data.aws_ami.ubuntu_22.id
  instance_type          = var.ec2_instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  key_name               = var.ec2_key_pair_name

  root_block_device {
    volume_size = var.ec2_volume_size
    volume_type = "gp3"
  }

  # Script chạy 1 lần khi EC2 boot lần đầu: cài Docker CE
  user_data = file("${path.module}/user_data.sh")

  tags = { Name = "${var.project_name}-backend" }
}

resource "aws_eip" "backend" {
  domain = "vpc"
  tags   = { Name = "${var.project_name}-eip" }
}

resource "aws_eip_association" "backend" {
  instance_id   = aws_instance.backend.id
  allocation_id = aws_eip.backend.id
}
