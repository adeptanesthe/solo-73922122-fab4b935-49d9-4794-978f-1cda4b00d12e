"""FastAPI 接口层自测（含参数校验与端到端调用）。"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(**overrides):
    base = {
        "power": [0.0] * 10 + [30.0] * 10 + [10.0] * 10,
        "noise_std": 0.8,
        "device1": {
            "name": "蒸饭柜",
            "on_power": 10.0,
            "off_power": 0.0,
            "min_on_minutes": 3,
            "min_off_minutes": 3,
        },
        "device2": {
            "name": "抽风机",
            "on_power": 20.0,
            "off_power": 0.0,
            "min_on_minutes": 3,
            "min_off_minutes": 3,
        },
        "confidence_threshold": 0.9,
    }
    base.update(overrides)
    return base


class TestApi(unittest.TestCase):
    def test_health(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_infer_end_to_end(self):
        r = client.post("/infer", json=_payload())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["window_minutes"], 30)
        self.assertEqual(data["missing_minutes"], 0)
        self.assertEqual(len(data["devices"]), 2)
        for dev in data["devices"]:
            self.assertEqual(len(dev["posterior_on"]), 30)
            covered = sum(i["duration_minutes"] for i in dev["intervals"])
            self.assertEqual(covered, 30)
            self.assertEqual(
                dev["running_minutes"] + dev["stopped_minutes"] + dev["uncertain_minutes"],
                30,
            )
            for iv in dev["intervals"]:
                self.assertIn(iv["state"], ("running", "stopped", "uncertain"))
                self.assertGreaterEqual(iv["confidence"], 0.0)

    def test_missing_values_accepted(self):
        power = [0.0] * 10 + [None] * 5 + [30.0] * 15
        r = client.post("/infer", json=_payload(power=power))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["missing_minutes"], 5)

    def test_threshold_must_be_above_half(self):
        r = client.post("/infer", json=_payload(confidence_threshold=0.5))
        self.assertEqual(r.status_code, 422)
        r = client.post("/infer", json=_payload(confidence_threshold=0.49))
        self.assertEqual(r.status_code, 422)

    def test_noise_std_positive(self):
        r = client.post("/infer", json=_payload(noise_std=0))
        self.assertEqual(r.status_code, 422)

    def test_minutes_range_validation(self):
        bad_device = {
            "name": "坏设备",
            "on_power": 1.0,
            "off_power": 0.0,
            "min_on_minutes": 0,
            "min_off_minutes": 31,
        }
        payload = _payload()
        payload["device1"] = bad_device
        r = client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 422)

    def test_window_too_long(self):
        r = client.post("/infer", json=_payload(power=[1.0] * 1441))
        self.assertEqual(r.status_code, 422)

    def test_duplicate_names_rejected(self):
        payload = _payload()
        payload["device2"]["name"] = payload["device1"]["name"]
        r = client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 422)

    def test_openapi_docs_available(self):
        r = client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/infer", r.json()["paths"])


if __name__ == "__main__":
    unittest.main()
