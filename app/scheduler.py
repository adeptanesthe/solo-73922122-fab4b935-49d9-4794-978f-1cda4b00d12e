"""恒功率设备错峰开机排程。

沿用 ``/infer`` 的总电表读数与双设备 HSMM：

* 窗口最后一分钟的后验是**增强状态** (s1, s2, c1, c2) 上的联合分布，
  其中 c1 / c2 是两台设备各自当前开/停状态已持续的分钟数。排程直接以该
  分布为起点，不把两台设备近似成相互独立，因此两台设备已持续时长（以及
  “还要多久才允许翻转”）被完整保留。
* 未来不再有观测，只按原转移表滚动。第 1 分钟是读数窗口结束后的下一分钟，
  排程视野固定为接下来的 60 分钟。
* 新设备在候选分钟开机后以恒定功率连续运行固定分钟数，其余分钟不贡献功率。
  每一分钟把“合计功率超过线路上限”的状态质量从存活路径中剔除。质量一旦
  被剔除就不再参与后续滚动，**同一条未来路径在多个分钟超限也只计一次**，
  最终 ``1 - 存活质量`` 即“至少一分钟超限”的并集概率。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .hsmm import DeviceParams, _Model, _prepare_inference

HORIZON_MINUTES = 60
PROBABILITY_DECIMALS = 6
# 仅用于吸收浮点误差，避免“概率恰好等于阈值”时被误判为超限。
_PROBABILITY_TOLERANCE = 1e-12


def _state_posterior(gamma: np.ndarray) -> np.ndarray:
    """窗口最后一分钟的增强状态联合后验（归一化）。"""

    last = np.exp(gamma[-1])
    total = float(last.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("窗口最后一分钟的后验分布无效")
    return last / total


def _linear_grouped_step(
    probs: np.ndarray,
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """概率空间的一步转移：target[j] = Σ_i probs[i] * P(i -> j)。"""

    out = np.zeros_like(probs)
    for _degree, (states, neighbours, log_weights) in groups.items():
        out[states] = (probs[neighbours] * np.exp(log_weights)).sum(axis=-1)
    return out


def evaluate_start(
    state_probs: np.ndarray,
    model: _Model,
    *,
    new_device_power: float,
    run_minutes: int,
    start_minute: int,
    line_power_limit: float,
    horizon_minutes: int = HORIZON_MINUTES,
) -> float:
    """计算一个候选开机分钟下的“任一分钟超限”概率。

    ``state_probs`` 是窗口最后一分钟的增强状态分布；``start_minute`` 为 1 基，
    即窗口结束后第 1 分钟开机。候选时段必须完整落在视野内
    （``start_minute + run_minutes - 1 <= horizon_minutes``）。
    """

    if not 1 <= start_minute <= horizon_minutes:
        raise ValueError("候选开机分钟必须位于未来视野内（从 1 开始）")
    if run_minutes < 1 or start_minute + run_minutes - 1 > horizon_minutes:
        raise ValueError("候选时段必须完整落在未来视野内")

    limit = float(line_power_limit)
    added = float(new_device_power)

    # 每一分钟的超限状态必须由完整增强状态判定：同一 (s1, s2) 但不同持续
    # 时长的状态功率相同，这里直接按状态功率比较即可。
    base_power = model.mean_power
    bad_when_running = (base_power + added) > limit
    bad_when_idle = base_power > limit

    probs = np.asarray(state_probs, dtype=np.float64)
    if probs.shape != (model.n_states,):
        raise ValueError("状态分布维度与模型不一致")
    if probs.min() < -_PROBABILITY_TOLERANCE:
        raise ValueError("状态分布不能含负概率")
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("状态分布概率之和必须为正")
    probs = probs / total

    end_minute = start_minute + run_minutes - 1
    for minute in range(1, horizon_minutes + 1):
        probs = _linear_grouped_step(probs, model.pred_groups)
        bad = bad_when_running if start_minute <= minute <= end_minute else bad_when_idle
        # 剔除已超限路径；其质量永久退出后续滚动，从而对并集只计一次。
        if bad.any():
            probs[bad] = 0.0
        alive = float(probs.sum())
        if alive <= 0.0:
            return 1.0

    probability = 1.0 - alive
    return float(min(max(probability, 0.0), 1.0))


def _inference_cache_key(
    power: tuple[float | None, ...],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> tuple:
    return (
        power,
        float(noise_std),
        (float(device1.on_power), float(device1.off_power), int(device1.min_on), int(device1.min_off)),
        (float(device2.on_power), float(device2.off_power), int(device2.min_on), int(device2.min_off)),
        float(switch_probability),
    )


@lru_cache(maxsize=16)
def _window_state_posterior_cached(
    power: tuple[float | None, ...],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> tuple[np.ndarray, _Model, int]:
    """推断窗口并返回末分钟增强状态后验；同参数重复排程时复用。"""

    gamma, _raw, model, _missing, _ll = _prepare_inference(
        list(power), noise_std, device1, device2, switch_probability
    )
    state_probs = _state_posterior(gamma)
    return state_probs, model, len(power)


def plan_schedule(
    power: list[float | None],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    *,
    new_device_power: float,
    run_minutes: int,
    candidate_start_minutes: list[int],
    line_power_limit: float,
    max_exceedance_probability: float,
    switch_probability: float = 0.1,
) -> dict:
    """评估所有候选开机分钟，返回最早满足超限概率上限的时段。

    候选分钟为 1 基，必须完整落在窗口结束后的 60 分钟内。返回的
    ``candidates`` 按开机分钟升序排列；``start_minute`` 为最早的合格候选。
    """

    if isinstance(run_minutes, bool) or not isinstance(run_minutes, int):
        raise ValueError("run_minutes 必须为整数")
    if not 1 <= run_minutes <= HORIZON_MINUTES:
        raise ValueError("run_minutes 必须位于 1 到 60 之间")
    if not 0.0 <= max_exceedance_probability <= 1.0:
        raise ValueError("max_exceedance_probability 必须位于 0 到 1 之间")
    for name, value in (
        ("new_device_power", new_device_power),
        ("line_power_limit", line_power_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须为数值")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} 必须为有限数值")
    if new_device_power < 0:
        raise ValueError("new_device_power 不能为负数")
    if not candidate_start_minutes:
        raise ValueError("candidate_start_minutes 不能为空")
    for start in candidate_start_minutes:
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError("候选开机分钟必须为整数")
        if start < 1 or start + run_minutes - 1 > HORIZON_MINUTES:
            raise ValueError(
                "候选时段必须完整落在窗口结束后的 60 分钟内"
                f"（开机分钟 1-{HORIZON_MINUTES - run_minutes + 1}）"
            )

    state_probs, model, window_minutes = _window_state_posterior_cached(
        tuple(power),
        float(noise_std),
        device1,
        device2,
        float(switch_probability),
    )

    evaluated: list[dict] = []
    for start in sorted(set(candidate_start_minutes)):
        probability = evaluate_start(
            state_probs,
            model,
            new_device_power=new_device_power,
            run_minutes=run_minutes,
            start_minute=start,
            line_power_limit=line_power_limit,
        )
        evaluated.append(
            {
                "start_minute": start,
                "end_minute": start + run_minutes - 1,
                "exceedance_probability": round(probability, PROBABILITY_DECIMALS),
                # 用未取整的真实概率判定，避免四舍五入把轻微超限变成合格。
                "within_limit": bool(
                    probability <= max_exceedance_probability + _PROBABILITY_TOLERANCE
                ),
            }
        )

    chosen = next((c for c in evaluated if c["within_limit"]), None)
    return {
        "window_minutes": window_minutes,
        "horizon_minutes": HORIZON_MINUTES,
        "run_minutes": run_minutes,
        "new_device_power": float(new_device_power),
        "line_power_limit": float(line_power_limit),
        "max_exceedance_probability": float(max_exceedance_probability),
        "found": chosen is not None,
        "start_minute": chosen["start_minute"] if chosen else None,
        "exceedance_probability": chosen["exceedance_probability"] if chosen else None,
        "message": "找到最早可用开机时段" if chosen else "无可用时段",
        "candidates": evaluated,
    }
