"""错峰开机排程逻辑自测：并集口径、时长保留与接口校验。"""

from __future__ import annotations

import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from app.hsmm import DeviceParams, _build_model
from app.main import app
from app.scheduler import (
    HORIZON_MINUTES,
    _linear_grouped_step,
    evaluate_start,
    plan_schedule,
)

client = TestClient(app)

D1 = DeviceParams(on_power=10.0, off_power=0.0, min_on=2, min_off=2)
D2 = DeviceParams(on_power=20.0, off_power=0.0, min_on=2, min_off=2)


def _uniform_state_probs(model) -> list[float]:
    return [1.0 / model.n_states] * model.n_states


def _brute_force_exceedance(
    model,
    start_probs: list[float],
    *,
    new_device_power: float,
    run_minutes: int,
    start_minute: int,
    line_power_limit: float,
    horizon_minutes: int,
) -> float:
    """穷举全部未来状态路径，按“路径并集”口径计算超限概率。

    不做任何状态聚合，每条路径单独展开，因此与滚动剔除法是相互独立的实现。
    仅用于很短的视野，避免路径数爆炸。
    """

    succ: list[list[tuple[int, float]]] = [[] for _ in range(model.n_states)]
    for degree, (states, neighbours, logw) in model.succ_groups.items():
        for row, src in enumerate(states):
            succ[int(src)] = [
                (int(neighbours[row, k]), float(math.exp(logw[row, k])))
                for k in range(degree)
            ]

    bad_running = model.mean_power + new_device_power > line_power_limit
    bad_idle = model.mean_power > line_power_limit
    end_minute = start_minute + run_minutes - 1

    exceed = 0.0
    # 初始分布对应窗口最后一分钟；未来第 1 分钟需先走一步转移再判超限。
    stack = [
        (s0, 0, start_probs[s0])
        for s0 in range(model.n_states)
        if start_probs[s0] > 0.0
    ]
    while stack:
        state, minute, weight = stack.pop()
        if minute >= 1:
            running = start_minute <= minute <= end_minute
            if (bad_running if running else bad_idle)[state]:
                exceed += weight
                continue
        if minute == horizon_minutes:
            continue
        for nxt, p in succ[state]:
            stack.append((nxt, minute + 1, weight * p))
    return exceed


class TestSchedulerMath(unittest.TestCase):
    def test_matches_brute_force_union_probability(self):
        # 短视野下逐路径穷举，验证“同一条路径多次超限只计一次”的并集口径。
        model = _build_model(D1, D2, 0.3)
        probs = _uniform_state_probs(model)
        for start, run, limit, add in [
            (1, 3, 25.0, 10.0),
            (2, 4, 30.0, 5.0),
            (4, 3, 20.0, 0.0),
            (1, 8, 40.0, 15.0),
            (5, 4, 35.0, 12.0),
        ]:
            with self.subTest(start=start, run=run, limit=limit, add=add):
                fast = evaluate_start(
                    probs,
                    model,
                    new_device_power=add,
                    run_minutes=run,
                    start_minute=start,
                    line_power_limit=limit,
                    horizon_minutes=8,
                )
                slow = _brute_force_exceedance(
                    model,
                    probs,
                    new_device_power=add,
                    run_minutes=run,
                    start_minute=start,
                    line_power_limit=limit,
                    horizon_minutes=8,
                )
                self.assertAlmostEqual(fast, slow, places=10)

    def test_repeated_violations_counted_once(self):
        # 新设备功率极高：一旦进入运行段必然超限，并集概率必须为 1，
        # 而不是把运行段每一分钟的超限概率累加（那会超过 1）。
        model = _build_model(D1, D2, 0.1)
        probs = _uniform_state_probs(model)
        p = evaluate_start(
            probs,
            model,
            new_device_power=1000.0,
            run_minutes=30,
            start_minute=1,
            line_power_limit=1.0,
            horizon_minutes=HORIZON_MINUTES,
        )
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_no_new_power_and_high_limit_never_exceeds(self):
        model = _build_model(D1, D2, 0.1)
        probs = _uniform_state_probs(model)
        p = evaluate_start(
            probs,
            model,
            new_device_power=0.0,
            run_minutes=10,
            start_minute=1,
            line_power_limit=31.0,
            horizon_minutes=HORIZON_MINUTES,
        )
        self.assertEqual(p, 0.0)

    def test_baseline_alone_over_limit_is_certain(self):
        # 上限低于最小值：无论新设备是否运行都必然超限。
        model = _build_model(D1, D2, 0.1)
        probs = _uniform_state_probs(model)
        p = evaluate_start(
            probs,
            model,
            new_device_power=5.0,
            run_minutes=1,
            start_minute=1,
            line_power_limit=-1.0,
            horizon_minutes=HORIZON_MINUTES,
        )
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_preserves_elapsed_durations_from_window_end(self):
        # 窗口末分钟两台设备都刚开机（持续 1 分钟），最短开机 2 分钟：
        # 未来第 1 分钟两台设备都不允许停机，支撑集必须仍为双开且计数为 2。
        model = _build_model(D1, D2, 0.5)
        idx = next(
            i
            for i in range(model.n_states)
            if model.state_s1[i] == 1
            and model.state_s2[i] == 1
            and model.state_c1[i] == 1
            and model.state_c2[i] == 1
        )
        probs = np.zeros(model.n_states)
        probs[idx] = 1.0
        nxt = _linear_grouped_step(probs, model.pred_groups)
        self.assertAlmostEqual(float(nxt.sum()), 1.0, places=12)
        support = np.flatnonzero(nxt > 1e-12)
        self.assertEqual(len(support), 1)
        j = int(support[0])
        self.assertEqual((int(model.state_s1[j]), int(model.state_s2[j])), (1, 1))
        self.assertEqual((int(model.state_c1[j]), int(model.state_c2[j])), (2, 2))

    def test_candidate_must_fit_inside_horizon(self):
        common = dict(
            power=[0.0, 10.0, 20.0],
            noise_std=1.0,
            device1=D1,
            device2=D2,
            new_device_power=5.0,
            line_power_limit=100.0,
            max_exceedance_probability=0.1,
        )
        with self.assertRaises(ValueError):
            plan_schedule(run_minutes=10, candidate_start_minutes=[52], **common)
        with self.assertRaises(ValueError):
            plan_schedule(run_minutes=51, candidate_start_minutes=[11], **common)
        with self.assertRaises(ValueError):
            plan_schedule(run_minutes=61, candidate_start_minutes=[1], **common)
        with self.assertRaises(ValueError):
            plan_schedule(run_minutes=5, candidate_start_minutes=[0], **common)
        with self.assertRaises(ValueError):
            plan_schedule(run_minutes=5, candidate_start_minutes=[], **common)

    def test_earliest_feasible_candidate_selected(self):
        # 窗口以“两台设备都停机”结束：越早开机越安全，最早候选胜出。
        power = [30.0] * 10 + [0.0] * 10
        out = plan_schedule(
            power,
            0.5,
            DeviceParams(10.0, 0.0, 2, 2),
            DeviceParams(20.0, 0.0, 2, 2),
            new_device_power=15.0,
            run_minutes=10,
            candidate_start_minutes=[30, 5, 20],
            line_power_limit=40.0,
            max_exceedance_probability=0.5,
        )
        self.assertTrue(out["found"])
        self.assertEqual([c["start_minute"] for c in out["candidates"]], [5, 20, 30])
        feasible = [c for c in out["candidates"] if c["within_limit"]]
        self.assertTrue(feasible)
        self.assertEqual(out["start_minute"], feasible[0]["start_minute"])
        self.assertEqual(
            out["exceedance_probability"], feasible[0]["exceedance_probability"]
        )
        # 更早的开机分钟不应比更晚的更危险（窗口末已停机）。
        probs = [c["exceedance_probability"] for c in out["candidates"]]
        self.assertEqual(probs, sorted(probs))

    def test_no_available_slot_returns_sentinel(self):
        # 上限压到低于任何组合功率：所有候选都必然超限。
        out = plan_schedule(
            [10.0] * 20,
            0.5,
            D1,
            D2,
            new_device_power=100.0,
            run_minutes=30,
            candidate_start_minutes=[1, 15, 31],
            line_power_limit=1.0,
            max_exceedance_probability=0.0,
        )
        self.assertFalse(out["found"])
        self.assertIsNone(out["start_minute"])
        self.assertIsNone(out["exceedance_probability"])
        self.assertEqual(out["message"], "无可用时段")
        self.assertTrue(
            all(c["exceedance_probability"] == 1.0 for c in out["candidates"])
        )
        self.assertTrue(all(not c["within_limit"] for c in out["candidates"]))

    def test_probabilities_stay_in_unit_interval(self):
        power = [0.0] * 5 + [20.0] * 5
        out = plan_schedule(
            power,
            2.0,
            D1,
            D2,
            new_device_power=12.0,
            run_minutes=20,
            candidate_start_minutes=list(range(1, 42)),
            line_power_limit=25.0,
            max_exceedance_probability=0.5,
        )
        self.assertEqual(len(out["candidates"]), 41)
        for c in out["candidates"]:
            self.assertGreaterEqual(c["exceedance_probability"], 0.0)
            self.assertLessEqual(c["exceedance_probability"], 1.0)

    def test_missing_minutes_supported_in_window(self):
        # 窗口含缺测分钟时排程仍可运行（缺测仍推进转移与时长）。
        out = plan_schedule(
            [0.0, None, 20.0, None, 10.0],
            1.0,
            D1,
            D2,
            new_device_power=5.0,
            run_minutes=5,
            candidate_start_minutes=[1],
            line_power_limit=60.0,
            max_exceedance_probability=1.0,
        )
        self.assertTrue(out["found"])
        self.assertEqual(out["start_minute"], 1)

    def test_full_day_window_runtime(self):
        import time

        power = [float((t * 7) % 31) for t in range(1440)]
        t0 = time.perf_counter()
        out = plan_schedule(
            power,
            3.0,
            DeviceParams(10.0, 0.0, 30, 30),
            DeviceParams(20.0, 0.0, 30, 30),
            new_device_power=12.0,
            run_minutes=30,
            candidate_start_minutes=list(range(1, 32)),
            line_power_limit=35.0,
            max_exceedance_probability=0.5,
        )
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(out["candidates"]), 31)
        self.assertLess(elapsed, 10.0, f"全天窗口排程耗时 {elapsed:.2f}s 过长")


class TestScheduleApi(unittest.TestCase):
    def _payload(self, **overrides):
        base = {
            "power": [30.0] * 10 + [0.0] * 10,
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
            "new_device_power": 15.0,
            "run_minutes": 20,
            "candidate_start_minutes": [1, 10, 20, 41],
            "line_power_limit": 40.0,
            "max_exceedance_probability": 0.5,
        }
        base.update(overrides)
        return base

    def test_schedule_end_to_end(self):
        r = client.post("/schedule", json=self._payload())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["window_minutes"], 20)
        self.assertEqual(data["horizon_minutes"], 60)
        self.assertEqual(len(data["candidates"]), 4)
        self.assertEqual(
            [c["start_minute"] for c in data["candidates"]], [1, 10, 20, 41]
        )
        if data["found"]:
            self.assertTrue(data["candidates"][0]["within_limit"])
            self.assertEqual(
                data["start_minute"], data["candidates"][0]["start_minute"]
            )
            self.assertLessEqual(
                data["exceedance_probability"], data["max_exceedance_probability"]
            )
        else:
            self.assertIsNone(data["start_minute"])
            self.assertIsNone(data["exceedance_probability"])
            self.assertEqual(data["message"], "无可用时段")

    def test_candidate_beyond_horizon_rejected(self):
        r = client.post(
            "/schedule",
            json=self._payload(run_minutes=20, candidate_start_minutes=[42]),
        )
        self.assertEqual(r.status_code, 422)

    def test_run_minutes_upper_bound(self):
        r = client.post("/schedule", json=self._payload(run_minutes=61))
        self.assertEqual(r.status_code, 422)

    def test_empty_candidates_rejected(self):
        r = client.post("/schedule", json=self._payload(candidate_start_minutes=[]))
        self.assertEqual(r.status_code, 422)

    def test_probability_threshold_range(self):
        r = client.post(
            "/schedule", json=self._payload(max_exceedance_probability=1.1)
        )
        self.assertEqual(r.status_code, 422)
        r = client.post(
            "/schedule", json=self._payload(max_exceedance_probability=-0.1)
        )
        self.assertEqual(r.status_code, 422)

    def test_negative_new_device_power_rejected(self):
        r = client.post("/schedule", json=self._payload(new_device_power=-1.0))
        self.assertEqual(r.status_code, 422)

    def test_openapi_lists_schedule(self):
        r = client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/schedule", r.json()["paths"])


if __name__ == "__main__":
    unittest.main()
