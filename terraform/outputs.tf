output "ec2_elastic_ip" {
  description = "Dán IP này vào fe/vercel.json và đặt là EC2_HOST trong GitHub Secrets"
  value       = aws_eip.backend.public_ip
}

output "rds_hostname" {
  description = "RDS hostname (dùng trong DATABASE_URL)"
  value       = aws_db_instance.postgres.address
  sensitive   = true
}

output "database_url" {
  description = "Full DATABASE_URL — copy vào /home/ubuntu/ocr-crnn/.env.prod trên EC2"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
  sensitive   = true
}

output "backend_api_url" {
  description = "URL của backend API (HTTP)"
  value       = "http://${aws_eip.backend.public_ip}:8000"
}

output "ssh_command" {
  description = "Lệnh SSH vào EC2"
  value       = "ssh -i ~/.ssh/ocr-crnn-keypair.pem ubuntu@${aws_eip.backend.public_ip}"
}

output "health_check_url" {
  description = "Kiểm tra backend đã chạy chưa"
  value       = "http://${aws_eip.backend.public_ip}:8000/health"
}
