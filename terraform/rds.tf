resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  tags       = { Name = "${var.project_name}-db-subnet-group" }
}

resource "aws_db_instance" "postgres" {
  identifier        = "${var.project_name}-db"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = var.rds_instance_class
  allocated_storage = 20
  storage_type      = "gp2"
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Không expose ra internet — chỉ EC2 trong cùng VPC mới kết nối được
  publicly_accessible = false

  # Backup 7 ngày
  backup_retention_period = 0
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # skip_final_snapshot = true để terraform destroy hoạt động clean
  # Đổi thành false và thêm final_snapshot_identifier nếu cần giữ data khi destroy
  skip_final_snapshot = true

  tags = { Name = "${var.project_name}-postgres" }
}
