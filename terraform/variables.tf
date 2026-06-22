variable "aws_region" {
  description = "AWS region (ap-southeast-1 = Singapore, latency thấp nhất từ VN)"
  type        = string
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Prefix cho tất cả AWS resource names"
  type        = string
  default     = "ocr-crnn"
}

variable "ec2_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "ec2_volume_size" {
  description = "Root EBS volume size (GB) — cần đủ cho Docker images (~2GB) + uploads"
  type        = number
  default     = 30
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "ocr_crnn"
}

variable "db_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "ocr"
}

variable "db_password" {
  description = "PostgreSQL master password — set trong terraform.tfvars (KHÔNG commit file này)"
  type        = string
  sensitive   = true
}

variable "ec2_key_pair_name" {
  description = "Tên EC2 key pair đã tạo trong AWS Console → EC2 → Key Pairs"
  type        = string
}

variable "my_ip_cidr" {
  description = "IP của bạn để restrict SSH, VD: 203.0.113.5/32. Tìm IP với: curl ifconfig.me"
  type        = string
  default     = "0.0.0.0/0"
}
