# Production HTTPS Setup (Domain + Let's Encrypt via Caddy)

This guide fixes mobile certificate warnings and enables live camera in browser.

## 1) Requirements

- A public domain (example: exam.yourdomain.com)
- A server with public IP
- Ports 80 and 443 open on firewall/security group
- Backend running on the same server (FastAPI on 127.0.0.1:8000)

## 2) DNS

Create DNS record:

- Type: A
- Host: exam (or your subdomain)
- Value: your server public IP

Wait until DNS is propagated.

## 3) Build frontend

Run on server:

```bash
cd /path/to/OCR_CRNN/fe
npm ci
npm run build
```

Frontend output folder:

- /path/to/OCR_CRNN/fe/dist

## 4) Run backend on localhost

Example (system service recommended):

```bash
cd /path/to/OCR_CRNN/be
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## 5) Install and configure Caddy

Use the example file:

- deploy/caddy/Caddyfile.example

Copy to /etc/caddy/Caddyfile and replace domain + frontend path.

Important lines:

- root * /var/www/ocr-crnn/fe/dist
- reverse_proxy 127.0.0.1:8000 for /api/* and /static/*

## 6) Start Caddy

```bash
sudo systemctl enable caddy
sudo systemctl restart caddy
sudo systemctl status caddy
```

Caddy will automatically request and renew Let's Encrypt certificates.

## 7) Test on phone

Open:

- https://exam.yourdomain.com

Expected:

- No certificate warning
- Camera live permission prompt appears normally

## 8) Notes for this project

- Frontend API config is now same-origin by default in production.
- You do NOT need to expose backend port 8000 publicly.
- Keep only 80/443 public.

## 9) Troubleshooting

- Still certificate warning:
  - Check DNS points to correct server
  - Ensure ports 80/443 are reachable from internet
  - Check Caddy logs: `journalctl -u caddy -f`
- API fails from frontend:
  - Confirm backend is running on 127.0.0.1:8000
  - Confirm Caddy has /api/* and /static/* reverse_proxy
- Camera still blocked:
  - Ensure URL is exactly https://... (not http://)
  - Recheck browser site permissions
