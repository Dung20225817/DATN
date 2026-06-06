#!/bin/sh
set -eu

python - <<'PY'
import os
import time

import psycopg2

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise SystemExit("DATABASE_URL is not set")

for attempt in range(1, 31):
    try:
        conn = psycopg2.connect(database_url)
        conn.close()
        break
    except Exception as exc:
        print(f"Waiting for database ({attempt}/30): {exc}", flush=True)
        time.sleep(2)
else:
    raise SystemExit("Database is not ready after 30 attempts")
PY

exec "$@"
