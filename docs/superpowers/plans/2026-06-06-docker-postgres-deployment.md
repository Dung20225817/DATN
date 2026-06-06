# Docker Postgres Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Docker deployment easy to initialize with a PostgreSQL container, persistent database storage, clear startup order, and practical deployment documentation.

**Architecture:** Keep PostgreSQL as a separate Docker Compose service, not inside the backend image. Backend connects through `DATABASE_URL` to the Compose service name `db`; PostgreSQL stores data in the named volume `pgdata`; uploaded/runtime files stay in the host bind mount `./storage`.

**Tech Stack:** Docker Compose, PostgreSQL 16 Alpine, FastAPI/Uvicorn backend, React/Vite frontend served by Caddy.

---

## Execution Notes

This plan was executed with subagent-driven checkpoints. During review, two corrections were added:

- Corrective Task 1A added the Docker build artifacts required by `docker-compose.yml`: backend Dockerfile, backend entrypoint, backend `.dockerignore`, frontend Dockerfile, frontend Caddyfile, and frontend `.dockerignore`.
- The frontend Caddy SPA fallback was moved into a final catch-all `handle` block so `/api/*` and `/static/*` are proxied before `try_files` runs.
- The external PostgreSQL runbook guidance was hardened to use a Compose override with `depends_on: !reset []` and `DATABASE_URL: ${DATABASE_URL}` instead of a misleading one-file Compose command.
- Final verification built the Docker images, started all services, checked backend health, checked frontend serving, checked Caddy API proxying, confirmed PostgreSQL tables, and stopped the stack while preserving the `pgdata` volume.

---

## File Structure

- Modify `docker-compose.yml`: add explicit backend dependency on the healthy PostgreSQL service, add configurable host ports, and keep DB internal to the Compose network.
- Modify `.env.docker.example`: document database credentials, host ports, and Google OAuth variables for first-run setup.
- Modify `README.md`: keep the short Docker quickstart and link to the detailed Docker deployment guide.
- Create `docs/DOCKER_DEPLOYMENT.md`: full Docker runbook including first run, verification, backup, restore, reset, logs, and external database mode.

---

### Task 1: Tighten Docker Compose Startup Order

**Files:**
- Modify: `docker-compose.yml`
- Test: Docker Compose config validation

- [ ] **Step 1: Update backend to wait on healthy database**

In `docker-compose.yml`, replace the backend service block with this version, preserving the existing `db` and `frontend` service names:

```yaml
  backend:
    build:
      context: ./be
    container_name: ocr-crnn-backend
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-ocr}:${POSTGRES_PASSWORD:-ocr_password}@db:5432/${POSTGRES_DB:-ocr_crnn}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID:-}
      PYTHONUNBUFFERED: "1"
    volumes:
      - ./storage:/workspace/storage
    ports:
      - "${BACKEND_PORT:-8000}:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"]
      interval: 15s
      timeout: 5s
      retries: 10
```

- [ ] **Step 2: Make frontend host port configurable**

In `docker-compose.yml`, replace the frontend `ports` block:

```yaml
    ports:
      - "${FRONTEND_PORT:-8080}:80"
```

- [ ] **Step 3: Validate Compose renders correctly**

Run:

```powershell
docker compose --env-file .env.docker.example config
```

Expected:

```text
services:
  backend:
    depends_on:
      db:
        condition: service_healthy
```

Also confirm rendered ports include `8000` for backend and `8080` for frontend when no custom port is set.

- [ ] **Step 4: Commit Task 1**

```powershell
git add docker-compose.yml
git commit -m "chore: clarify docker service startup order"
```

---

### Task 2: Improve Docker Environment Template

**Files:**
- Modify: `.env.docker.example`
- Test: Docker Compose config validation

- [ ] **Step 1: Replace `.env.docker.example` contents**

Use this exact content:

```dotenv
# Compose project name controls container/volume/network prefixes.
COMPOSE_PROJECT_NAME=ocr_crnn

# PostgreSQL container initialization.
# Change POSTGRES_PASSWORD before deploying outside local demo.
POSTGRES_DB=ocr_crnn
POSTGRES_USER=ocr
POSTGRES_PASSWORD=ocr_password

# Host ports.
FRONTEND_PORT=8080
BACKEND_PORT=8000

# Google Sign-In.
# Use the same Web OAuth client ID for backend verification and frontend login.
GOOGLE_CLIENT_ID=your_google_web_client_id.apps.googleusercontent.com
VITE_GOOGLE_CLIENT_ID=your_google_web_client_id.apps.googleusercontent.com
```

- [ ] **Step 2: Validate environment values are consumed**

Run:

```powershell
docker compose --env-file .env.docker.example config
```

Expected:

```text
POSTGRES_DB: ocr_crnn
POSTGRES_PASSWORD: ocr_password
POSTGRES_USER: ocr
DATABASE_URL: postgresql://ocr:ocr_password@db:5432/ocr_crnn
published: "8000"
published: "8080"
```

- [ ] **Step 3: Commit Task 2**

```powershell
git add .env.docker.example
git commit -m "docs: document docker environment defaults"
```

---

### Task 3: Add Docker Deployment Runbook

**Files:**
- Create: `docs/DOCKER_DEPLOYMENT.md`
- Test: Read through commands against current Compose file

- [ ] **Step 1: Create `docs/DOCKER_DEPLOYMENT.md`**

Use this exact content:

```markdown
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
```

- [ ] **Step 2: Commit Task 3**

```powershell
git add docs/DOCKER_DEPLOYMENT.md
git commit -m "docs: add docker deployment runbook"
```

---

### Task 4: Update README Docker Section

**Files:**
- Modify: `README.md`
- Test: Confirm Docker section points to the runbook

- [ ] **Step 1: Replace the current Docker section with a short quickstart**

Replace the `## Chay bang Docker` section in `README.md` with this content, keeping the existing Vietnamese heading style if the file encoding displays correctly in the editor:

```markdown
## Chay bang Docker

Docker Compose chay 3 service:

- `db`: PostgreSQL 16, luu du lieu trong volume `pgdata`.
- `backend`: FastAPI, tu tao bang khi khoi dong neu database trong.
- `frontend`: Caddy phuc vu React build va reverse proxy `/api/*`, `/static/*` ve backend.

```powershell
Copy-Item .env.docker.example .env.docker
docker compose --env-file .env.docker up --build
```

Sau khi chay:

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`
- Health check: `http://localhost:8000/health`

Huong dan chi tiet ve cau hinh, backup, restore, reset du lieu va dung database ben ngoai nam trong `docs/DOCKER_DEPLOYMENT.md`.
```

- [ ] **Step 2: Confirm README no longer implies PostgreSQL is embedded in backend**

Run:

```powershell
rg -n "PostgreSQL|Docker|DOCKER_DEPLOYMENT" README.md
```

Expected:

```text
README.md:...:Docker Compose chay 3 service:
README.md:...:`db`: PostgreSQL 16, luu du lieu trong volume `pgdata`.
README.md:...:docs/DOCKER_DEPLOYMENT.md
```

- [ ] **Step 3: Commit Task 4**

```powershell
git add README.md
git commit -m "docs: simplify docker quickstart"
```

---

### Task 5: Verify Full Docker Deployment

**Files:**
- No file changes
- Test: Docker build and runtime health checks

- [ ] **Step 1: Validate Compose before building**

Run:

```powershell
docker compose --env-file .env.docker.example config
```

Expected: command exits with code `0`.

- [ ] **Step 2: Build images**

Run:

```powershell
docker compose --env-file .env.docker.example build
```

Expected: backend and frontend images build successfully. The backend image install should not install PyTorch, Transformers, EasyOCR, VietOCR, or OpenAI SDK because they are not in `be/requirements.txt`.

- [ ] **Step 3: Start stack detached**

Run:

```powershell
docker compose --env-file .env.docker.example up -d
```

Expected: all three services start.

- [ ] **Step 4: Check service status**

Run:

```powershell
docker compose --env-file .env.docker.example ps
```

Expected:

```text
ocr-crnn-db        ... healthy
ocr-crnn-backend   ... healthy
ocr-crnn-frontend  ... running
```

- [ ] **Step 5: Check backend health**

Run:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

Expected response:

```text
StatusCode        : 200
```

Response body:

```json
{"status":"healthy"}
```

- [ ] **Step 6: Check frontend is served**

Run:

```powershell
Invoke-WebRequest http://localhost:8080
```

Expected response:

```text
StatusCode        : 200
```

Body should contain the built React app HTML.

- [ ] **Step 7: Check database schema exists**

Run:

```powershell
docker compose --env-file .env.docker.example exec db psql -U ocr -d ocr_crnn -c "\dt"
```

Expected output includes:

```text
omr_assignment
omr_grade_result
omr_test
users
```

- [ ] **Step 8: Stop stack without deleting data**

Run:

```powershell
docker compose --env-file .env.docker.example down
```

Expected: containers stop; named volume `ocr_crnn_pgdata` remains.

- [ ] **Step 9: Commit verification notes only if a file was added**

If verification revealed documentation corrections, commit those corrections:

```powershell
git add README.md docs/DOCKER_DEPLOYMENT.md docker-compose.yml .env.docker.example
git commit -m "docs: finalize docker deployment verification"
```

If no file changed during verification, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers the approved approach of using PostgreSQL in Docker Compose for easy initialization, keeping DB separate from backend, persisting DB data in `pgdata`, and documenting external database mode.
- Placeholder scan: No unfinished marker words or unspecified implementation steps remain.
- Type/config consistency: Service names are consistently `db`, `backend`, and `frontend`; database name/user/password match `.env.docker.example`; backend URL uses `db:5432`; frontend and backend host ports match the documented defaults.
