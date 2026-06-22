terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State lưu trên S3 để GitHub Actions có thể đọc/ghi giữa các lần chạy.
  # Bucket phải tạo thủ công trước trong AWS Console (xem checklist Bước 1).
  backend "s3" {
    bucket = "ocr-crnn-terraform-state"
    key    = "terraform.tfstate"
    region = "ap-southeast-1"
  }
}

provider "aws" {
  region = var.aws_region
}
