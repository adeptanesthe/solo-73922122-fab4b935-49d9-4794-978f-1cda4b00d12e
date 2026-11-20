"""核心推断逻辑的关键功能自测。

运行： python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from app.hsmm import DeviceParams, infer_posterior, label_intervals, _build_model

D1 = DeviceParams(on_power=10.0, off_power=0.0, min_on=5, min_off=5)
D2 = DeviceParams(on_power=20.0, off_power=0.0, min_on=3, min_off=3)


def make_series(s1: list[int], s2: list[int], noise_std: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    t_total = len(s1)
    y = np.empty(t_total)
    for t in range(t_total):
        mu = (D1.on_power if s1[t] else D1.off_power) + (
            D2.on_power if s2[t] else D2.off_power
        )
        y[t] = mu + rng.normal(0.0, noise_std)
    return y.tolist()


class TestModelMechanics(unittest.TestCase):
    def test_clean_recovery(self):
        # 低噪声且噪声尺度与模型一致：四个组合功率 0/10/20/30 完全可分，
        # 应准确恢复真实状态。
        s1 = [0] * 10 + [1] * 20 + [0] * 10
        s2 = [1] * 15 + [0] * 25
        power = make_series(s1, s2, noise_std=0.5, seed=1)
        out = infer_posterior(power, 0.5, D1, D2)
        on1, on2 = out["posterior_on"]
        for t, p in enumerate(on1):
            self.assertAlmostEqual(p, float(s1[t]), places=3, msg=f"d1 t={t}")
        for t, p in enumerate(on2):
            self.assertAlmostEqual(p, float(s2[t]), places=3, msg=f"d2 t={t}")

    def test_transition_distributions_normalized(self):
        # 每个源状态的后继概率之和必须为 1。
        m = _build_model(D1, D2, 0.1)
        for d, (states, _idx, logw) in m.succ_groups.items():
            sums = np.exp(logw).sum(axis=1)
            np.testing.assert_allclose(
                sums, np.ones(len(states)), rtol=1e-12,
                err_msg=f"出度 {d} 分组转移概率未归一化",
            )
        # 前驱/后继定长表与分组表的状态总数一致
        self.assertEqual(m.succ_idx_fixed.shape[0], m.n_states)

    def test_min_duration_enforced_in_transitions(self):
        # 最短时长约束应直接体现在转移表上：未满足当前状态最短时长的
        # 增强状态，其任何后继都不允许翻转对应设备。
        dev1 = DeviceParams(10.0, 0.0, 8, 8)
        dev2 = DeviceParams(20.0, 0.0, 5, 7)
        m = _build_model(dev1, dev2, 0.1)
        for i in range(m.n_states):
            s1, s2 = int(m.state_s1[i]), int(m.state_s2[i])
            c1, c2 = int(m.state_c1[i]), int(m.state_c2[i])
            for slot in m.succ_idx_fixed[i]:
                if slot < 0:
                    continue
                ns1, ns2 = int(m.state_s1[slot]), int(m.state_s2[slot])
                if ns1 != s1:
                    self.assertGreaterEqual(c1, 8, "设备1提前翻转")
                if ns2 != s2:
                    need = 5 if s2 == 1 else 7
                    self.assertGreaterEqual(c2, need, "设备2提前翻转")

    def test_min_duration_shapes_posterior(self):
        # 端到端：清晰信号下输出的高置信内部开/停段不短于最短时长。
        dev1 = DeviceParams(10.0, 0.0, 8, 8)
        dev2 = DeviceParams(20.0, 0.0, 8, 8)
        s1 = [0] * 16 + [1] * 16 + [0] * 16
        s2 = [1] * 16 + [0] * 16 + [1] * 16
        power = make_series(s1, s2, noise_std=1.0, seed=2)
        out = infer_posterior(power, 1.0, dev1, dev2)
        for posterior in out["posterior_on"]:
            intervals = label_intervals(
                posterior, [False] * len(posterior), 0.9
            )
            for iv in intervals:
                if iv["state"] == "uncertain" or iv["truncated"]:
                    continue
                self.assertGreaterEqual(iv["duration_minutes"], 8)

    def test_missing_minutes_still_count_in_duration(self):
        s1 = [0] * 10 + [1] * 20 + [0] * 10
        s2 = [0] * 40
        power = make_series(s1, s2, noise_std=0.5, seed=3)
        # 中间缺测 8 分钟
        missing_idx = set(range(15, 23))
        power_with_holes = [None if t in missing_idx else v for t, v in enumerate(power)]
        out = infer_posterior(power_with_holes, 0.5, D1, D2)
        intervals = label_intervals(
            out["posterior_on"][0], out["missing"], 0.9
        )
        # 所有区间长度之和必须等于完整窗口 40（缺测不丢分钟）
        total = sum(iv["duration_minutes"] for iv in intervals)
        self.assertEqual(total, 40)
        # 缺测分钟必须出现在某个区间的 missing_minutes 计数里
        self.assertEqual(
            sum(iv["missing_minutes"] for iv in intervals), len(missing_idx)
        )
        # 缺测区位于开机段内，两侧证据充分，模型应保持开机判断
        for t in range(15, 23):
            self.assertGreater(out["posterior_on"][0][t], 0.75)

    def test_boundary_truncation_flag(self):
        s1 = [1] * 30 + [0] * 10
        s2 = [0] * 40
        power = make_series(s1, s2, noise_std=0.5, seed=4)
        out = infer_posterior(power, 0.5, D1, D2)
        intervals = label_intervals(
            out["posterior_on"][0], [False] * 40, 0.95
        )
        first, last = intervals[0], intervals[-1]
        self.assertEqual(first["state"], "running")
        self.assertIn("start", first["truncated_edges"])
        self.assertTrue(first["truncated"])
        self.assertEqual(last["state"], "stopped")
        self.assertIn("end", last["truncated_edges"])

    def test_low_confidence_goes_to_uncertain(self):
        # 极低信噪比：功率组合互相接近 + 大噪声，严格阈值下应产生 uncertain，
        # 且阈值越严格 uncertain 越多。
        dev1 = DeviceParams(on_power=1.0, off_power=0.0, min_on=2, min_off=2)
        dev2 = DeviceParams(on_power=1.0, off_power=0.0, min_on=2, min_off=2)
        s1 = [0] * 8 + [1] * 8 + [0] * 8
        s2 = [1] * 24
        power = make_series(s1, s2, noise_std=5.0, seed=5)
        out = infer_posterior(power, 5.0, dev1, dev2)
        iv_loose = label_intervals(out["posterior_on"][0], [False] * 24, 0.55)
        iv_strict = label_intervals(out["posterior_on"][0], [False] * 24, 0.99)
        unc_loose = sum(i["duration_minutes"] for i in iv_loose if i["state"] == "uncertain")
        unc_strict = sum(i["duration_minutes"] for i in iv_strict if i["state"] == "uncertain")
        self.assertGreaterEqual(unc_strict, unc_loose)
        self.assertGreater(unc_strict, 0)

    def test_threshold_must_exceed_half(self):
        posterior = [0.9] * 5
        for bad in (0.5, 0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                label_intervals(posterior, [False] * 5, bad)

    def test_posteriors_are_valid_probabilities(self):
        s1 = [0] * 12 + [1] * 12
        s2 = [1] * 24
        power = make_series(s1, s2, noise_std=1.5, seed=6)
        out = infer_posterior(power, 1.5, D1, D2)
        for posterior in out["posterior_on"]:
            self.assertEqual(len(posterior), 24)
            for p in posterior:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
        self.assertTrue(math.isfinite(out["log_likelihood"]))

    def test_all_missing_returns_prior(self):
        # 全部缺测：似然恒定，后验为先验下的有限边际概率。
        out = infer_posterior([None] * 30, 1.0, D1, D2)
        for posterior in out["posterior_on"]:
            self.assertEqual(len(posterior), 30)
            for p in posterior:
                self.assertTrue(0.0 < p < 1.0)
                self.assertTrue(math.isfinite(p))

    def test_window_length_validation(self):
        with self.assertRaises(ValueError):
            infer_posterior([], 1.0, D1, D2)
        with self.assertRaises(ValueError):
            infer_posterior([1.0] * 1441, 1.0, D1, D2)

    def test_identifiability_ambiguity_raises_uncertainty(self):
        # 两台设备开机功率相同：只开一台时无法区分是谁，
        # 对应分钟两台设备的开机后验都不应达到高置信（结构性歧义 -> 待核实）。
        dev1 = DeviceParams(on_power=10.0, off_power=0.0, min_on=3, min_off=3)
        dev2 = DeviceParams(on_power=10.0, off_power=0.0, min_on=3, min_off=3)
        s1 = [1] * 12 + [0] * 8
        s2 = [0] * 20
        rng = np.random.default_rng(7)
        power = [
            (10.0 if s1[t] else 0.0) + rng.normal(0.0, 0.5)
            for t in range(20)
        ]
        out = infer_posterior(power, 0.5, dev1, dev2)
        iv1 = label_intervals(out["posterior_on"][0], [False] * 20, 0.95)
        iv2 = label_intervals(out["posterior_on"][1], [False] * 20, 0.95)
        self.assertFalse(any(i["state"] == "running" for i in iv1))
        self.assertFalse(any(i["state"] == "running" for i in iv2))

    def test_full_day_runtime_performance(self):
        # 1440 分钟 + 最长最短时长（30）：状态空间最大，验证可在合理时间完成。
        dev1 = DeviceParams(10.0, 0.0, 30, 30)
        dev2 = DeviceParams(20.0, 0.0, 30, 30)
        rng = np.random.default_rng(8)
        power = [float(rng.normal(0.0, 1.0)) for _ in range(1440)]
        import time

        t0 = time.perf_counter()
        out = infer_posterior(power, 1.0, dev1, dev2)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(out["posterior_on"][0]), 1440)
        self.assertLess(elapsed, 10.0, f"全天推断耗时 {elapsed:.2f}s 过长")


if __name__ == "__main__":
    unittest.main()
