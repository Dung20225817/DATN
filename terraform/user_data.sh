#!/bin/bash
# Script này chạy tự động khi EC2 khởi động lần đầu (user_data).
# Log tại /var/log/user-data.log để debug nếu cần.
set -euo pipefail
exec >> /var/log/user-data.log 2>&1

echo "[$(date)] Starting user_data setup..."

# System update
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Cài Docker CE từ official repository (không dùng docker.io của Ubuntu APT)
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

# Bật Docker và thêm user ubuntu vào group docker
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Cài thêm tiện ích hữu dụng
apt-get install -y htop curl wget unzip

# Tạo thư mục storage khớp với bind-mount trong docker-compose.prod.yml
# be/app/core/paths.py resolve STORAGE_ROOT = /workspace/storage
APP_DIR=/home/ubuntu/ocr-crnn
mkdir -p "${APP_DIR}/storage/uploads/omr"
mkdir -p "${APP_DIR}/storage/uploads/answer_keys"
mkdir -p "${APP_DIR}/storage/uploads/omr_data"
mkdir -p "${APP_DIR}/storage/uploads/omr_templates"
mkdir -p "${APP_DIR}/storage/uploads/temp"
mkdir -p "${APP_DIR}/storage/logs"
chown -R ubuntu:ubuntu "${APP_DIR}"

echo "[$(date)] user_data complete."
echo "[$(date)] Docker version: $(docker --version)"
echo "[$(date)] Docker Compose version: $(docker compose version)"
