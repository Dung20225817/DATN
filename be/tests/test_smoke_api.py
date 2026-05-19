import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        sqlite_path = Path(cls._tmpdir.name) / "smoke.sqlite3"
        os.environ["DATABASE_URL"] = f"sqlite:///{sqlite_path.as_posix()}"

        from main import app  # noqa: WPS433 - env must be configured before app import

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        from app.db.session import engine

        cls.client.close()
        engine.dispose()
        cls._tmpdir.cleanup()

    def test_health_and_router_registration(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")

        paths = {route.path for route in self.client.app.routes}
        expected_paths = {
            "/api/login",
            "/api/register",
            "/api/omr/form-profiles",
            "/api/omr/form-samples",
            "/api/omr/assignments",
            "/api/omr/assignments/{uid}",
            "/api/omr/assignments/{uid}/{aid}",
            "/api/omr/grade",
            "/api/omr/grade-batch",
            "/api/omr/suggest-crop",
        }
        self.assertTrue(expected_paths.issubset(paths))

    def test_auth_and_assignment_crud(self):
        register = self.client.post(
            "/api/register",
            json={
                "user_name": "Smoke Test",
                "email": "smoke@example.com",
                "phone": "0123456789",
                "password": "secret",
            },
        )
        self.assertEqual(register.status_code, 200)
        uid = register.json()["uid"]

        login = self.client.post(
            "/api/login",
            json={"email": "smoke@example.com", "password": "secret"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["uid"], uid)

        create = self.client.post(
            "/api/omr/assignments",
            data={
                "uid": str(uid),
                "title": "Smoke assignment",
                "question_count": "20",
                "total_points": "10",
            },
        )
        self.assertEqual(create.status_code, 200)
        assignment = create.json()["assignment"]
        aid = assignment["aid"]

        listed = self.client.get(f"/api/omr/assignments/{uid}")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["assignments"]), 1)

        update = self.client.put(
            f"/api/omr/assignments/{uid}/{aid}",
            data={
                "title": "Smoke assignment updated",
                "answer_sets": '[{"code":"001","answers":["A","B"]}]',
                "active_code": "001",
            },
        )
        self.assertEqual(update.status_code, 200)
        self.assertEqual(update.json()["assignment"]["title"], "Smoke assignment updated")

        delete = self.client.delete(f"/api/omr/assignments/{uid}/{aid}")
        self.assertEqual(delete.status_code, 200)

        listed_after_delete = self.client.get(f"/api/omr/assignments/{uid}")
        self.assertEqual(listed_after_delete.status_code, 200)
        self.assertEqual(listed_after_delete.json()["assignments"], [])

    def test_profile_listing_endpoints(self):
        profiles = self.client.get("/api/omr/form-profiles")
        self.assertEqual(profiles.status_code, 200)
        self.assertIn("profiles", profiles.json())

        samples = self.client.get("/api/omr/form-samples")
        self.assertEqual(samples.status_code, 200)
        self.assertIn("samples", samples.json())
