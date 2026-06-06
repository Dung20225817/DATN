# Docker Deployment Guide

This project can run as three Docker Compose services:

- `db`: PostgreSQL 16, stores data in the named volume `pgdata`.
- `backend`: FastAPI API, connects to PostgreSQL through `DATABASE_URL`.
- `frontend`: Caddy static server for the Vite build, reverse-proxies `/api/*` and `/static/*` to backend.

PostgreSQL should stay in its own container. Do not install PostgreSQL inside the backend image.

## First Run

```powershell
Copy-Item .env.docker.example .env.docker
```

Edit `.env.docker` before production use:

- Change `POSTGRES_PASSWORD`.
- Set `GOOGLE_CLIENT_ID` and `VITE_GOOGLE_CLIENT_ID` if Google login is used.
- Change `FRONTEND_PORT` or `BACKEND_PORT` if those host ports are already used.

Start the stack:

```powershell
docker compose --env-file .env.docker up --build
```

Open:

- Frontend: `http://localhost:8080`
- Backend health check: `http://localhost:8000/health`

## Why PostgreSQL Is In Compose

Keeping PostgreSQL in Compose is useful for local, demo, and small VPS deployment because a fresh machine can start the full stack with one command. The database data is not stored inside the container filesystem; it is stored in the named Docker volume `pgdata`, so restarting or recreating containers keeps the data.

For a larger production deployment, use an external or managed PostgreSQL instance and point `DATABASE_URL` to that database.

## Verify Services

```powershell
docker compose --env-file .env.docker ps
```

Expected service state:

- `ocr-crnn-db`: healthy
- `ocr-crnn-backend`: healthy
- `ocr-crnn-frontend`: running

Check backend:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

Expected response body:

```json
{"status":"healthy"}
```

Check database tables:

```powershell
docker compose --env-file .env.docker exec db psql -U ocr -d ocr_crnn -c "\dt"
```

Expected: tables such as `users`, `omr_assignment`, `omr_test`, and `omr_grade_result`.

## Logs

```powershell
docker compose --env-file .env.docker logs backend
docker compose --env-file .env.docker logs db
docker compose --env-file .env.docker logs frontend
```

Backend startup should include:

```text
Database tables initialized
All routers registered successfully
```

## Stop And Restart

Stop containers without deleting data:

```powershell
docker compose --env-file .env.docker down
```

Start again:

```powershell
docker compose --env-file .env.docker up -d
```

## Backup PostgreSQL

```powershell
docker compose --env-file .env.docker exec db pg_dump -U ocr -d ocr_crnn > backup_ocr_crnn.sql
```

## Restore PostgreSQL

Use this only with a database that is ready to receive the restore.

```powershell
Get-Content backup_ocr_crnn.sql | docker compose --env-file .env.docker exec -T db psql -U ocr -d ocr_crnn
```

## Reset Local Demo Data

This deletes PostgreSQL data. Use it only for local/demo reset.

```powershell
docker compose --env-file .env.docker down -v
docker compose --env-file .env.docker up --build
```

## Use External PostgreSQL

For production with an external database, set `DATABASE_URL` directly for the backend service or add it to an override Compose file.

Example backend environment:

```yaml
environment:
  DATABASE_URL: postgresql://user:password@db-host:5432/ocr_crnn
  GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID:-}
  PYTHONUNBUFFERED: "1"
```

Then run only backend and frontend if the Compose `db` service is not needed:

```powershell
docker compose --env-file .env.docker up --build backend frontend
```
