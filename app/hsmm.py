"""联合推断两台二进制设备开停状态的概率模型。

模型为显式时长隐半马尔可夫模型 (HSMM)：

* 隐状态为两台设备状态的笛卡尔积 (s1, s2)，各自带一个“当前状态已持续分钟数”
  计数器 (c1, c2)，计数器达到上限后封顶，从而保证任意开/停段长度都可表达。
* 只有在当前状态持续时间达到该状态的最短时长后，设备才允许翻转；
  翻转后计数器重置为 1，否则计数器递增（封顶）。
* 观测为两台设备各自功率之和叠加高斯噪声 N(0, sigma^2)；
  缺测分钟不产生观测似然（对所有状态似然为 1），但状态转移照常进行，
  因此缺测分钟仍计入开/停时长。
* 允许状态在观测窗口首尾截断：初始分布对所有增强状态均匀，
  不要求第一个状态已经持续满最短时长。

推理使用对数空间前向-后向算法，得到每个分钟各设备处于开机状态的
后验概率，再按置信度阈值映射为 running / stopped / uncertain。

转移按入度/出度分组成紧凑数组，绝大多数状态只有 1 个前驱/后继，
从而让全天 1440 分钟的推断保持在秒级。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NEG_INF = -math.inf
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class DeviceParams:
    """单台设备的功率与最短时长参数。"""

    on_power: float
    off_power: float
    min_on: int
    min_off: int


@dataclass
class _Model:
    """编译后的增强状态空间与分组转移结构。

    pred_groups / succ_groups: 键为入度/出度 d，值为
    (目标状态数组, 对应的 d 个邻居索引数组, 对应对数权重数组)。
    """

    n_states: int
    k1: int
    k2: int
    state_s1: np.ndarray
    state_s2: np.ndarray
    state_c1: np.ndarray
    state_c2: np.ndarray
    mean_power: np.ndarray
    pred_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    succ_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    # 定长表仅用于模型结构检查（测试），推断路径使用分组表。
    succ_idx_fixed: np.ndarray
    pred_idx_fixed: np.ndarray


def _group_edges(
    edges: list[list[tuple[int, float]]],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把每个状态的邻居列表按度分组成紧凑数组。"""

    degrees = np.array([len(e) for e in edges], dtype=np.int32)
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for d in np.unique(degrees):
        states = np.where(degrees == d)[0].astype(np.int32)
        idx = np.empty((len(states), int(d)), dtype=np.int32)
        logw = np.empty((len(states), int(d)), dtype=np.float64)
        for row, st in enumerate(states):
            nb = edges[st]
            idx[row] = [j for j, _ in nb]
            logw[row] = [lw for _, lw in nb]
        groups[int(d)] = (states, idx, logw)
    return groups


def _build_model(
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float,
) -> _Model:
    """枚举增强状态并构造前驱/后继转移表。"""

    k1 = max(device1.min_on, device1.min_off)
    k2 = max(device2.min_on, device2.min_off)

    states: list[tuple[int, int, int, int]] = []
    for s1 in (0, 1):
        for s2 in (0, 1):
            for c1 in range(1, k1 + 1):
                for c2 in range(1, k2 + 1):
                    states.append((s1, s2, c1, c2))

    n_states = len(states)
    index = {st: i for i, st in enumerate(states)}
    s1_arr = np.array([st[0] for st in states], dtype=np.int8)
    s2_arr = np.array([st[1] for st in states], dtype=np.int8)
    c1_arr = np.array([st[2] for st in states], dtype=np.int8)
    c2_arr = np.array([st[3] for st in states], dtype=np.int8)

    power1 = np.where(s1_arr == 1, device1.on_power, device1.off_power)
    power2 = np.where(s2_arr == 1, device2.on_power, device2.off_power)
    mean_power = power1 + power2

    p = float(switch_probability)

    successors: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]

    for i, (s1, s2, c1, c2) in enumerate(states):
        can_flip_1 = c1 >= (device1.min_on if s1 == 1 else device1.min_off)
        can_flip_2 = c2 >= (device2.min_on if s2 == 1 else device2.min_off)
        options: list[tuple[bool, bool]] = [(False, False)]
        if can_flip_1:
            options.append((True, False))
        if can_flip_2:
            options.append((False, True))
        if can_flip_1 and can_flip_2:
            options.append((True, True))

        weighted: list[tuple[int, float]] = []
        weight_sum = 0.0
        for flip1, flip2 in options:
            ns1 = 1 - s1 if flip1 else s1
            ns2 = 1 - s2 if flip2 else s2
            nc1 = 1 if flip1 else min(c1 + 1, k1)
            nc2 = 1 if flip2 else min(c2 + 1, k2)
            j = index[(ns1, ns2, nc1, nc2)]
            n_flips = int(flip1) + int(flip2)
            # 两台设备独立翻转概率的乘积，随后在合法选项间归一化，
            # 使每个源状态的条件转移分布之和为 1。
            w = (p ** n_flips) * ((1.0 - p) ** (2 - n_flips))
            weighted.append((j, w))
            weight_sum += w
        for j, w in weighted:
            successors[i].append((j, math.log(w / weight_sum)))

    predecessors: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
    for i, edges in enumerate(successors):
        for j, lw in edges:
            predecessors[j].append((i, lw))

    succ_groups = _group_edges(successors)
    pred_groups = _group_edges(predecessors)

    max_succ = max(len(e) for e in successors)
    max_pred = max(len(e) for e in predecessors)
    assert max_succ <= 4 and max_pred <= 8

    succ_fixed = np.full((n_states, 4), -1, dtype=np.int32)
    pred_fixed = np.full((n_states, 8), -1, dtype=np.int32)
    for i, edges in enumerate(successors):
        for slot, (j, _) in enumerate(edges):
            succ_fixed[i, slot] = j
    for i, edges in enumerate(predecessors):
        for slot, (j, _) in enumerate(edges):
            pred_fixed[i, slot] = j

    return _Model(
        n_states=n_states,
        k1=k1,
        k2=k2,
        state_s1=s1_arr,
        state_s2=s2_arr,
        state_c1=c1_arr,
        state_c2=c2_arr,
        mean_power=mean_power,
        pred_groups=pred_groups,
        succ_groups=succ_groups,
        succ_idx_fixed=succ_fixed,
        pred_idx_fixed=pred_fixed,
    )


def _log_emission(
    power: np.ndarray,
    missing: np.ndarray,
    model: _Model,
    noise_std: float,
) -> np.ndarray:
    """返回 (T, S) 观测对数似然；缺测行全为 0。"""

    t_total = power.shape[0]
    emit = np.zeros((t_total, model.n_states), dtype=np.float64)
    observed = ~missing
    if not observed.any():
        return emit

    y = power[observed][:, None]  # (O, 1)
    mu = model.mean_power[None, :]  # (1, S)
    z = (y - mu) / noise_std
    emit[observed] = -0.5 * z * z - math.log(noise_std) - LOG_SQRT_2PI
    return emit


def _grouped_logsumexp(
    values_prev: np.ndarray,
    emit_row: np.ndarray | None,
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """按度分组计算 logsumexp。

    values_prev: 上一时刻（前向）或当前时刻（后向）的 (S,) 对数值。
    emit_row: 后向时需叠加的当前时刻发射对数似然 (S,)；前向传 None。
    返回与每个状态对齐的 (S,) 结果。
    """

    out = np.full(values_prev.shape[0], NEG_INF, dtype=np.float64)
    for d, (states, nb_idx, logw) in groups.items():
        gathered = values_prev[nb_idx] + logw  # (n_d, d)
        if emit_row is not None:
            gathered = gathered + emit_row[nb_idx]
        row_max = gathered.max(axis=-1)
        out[states] = row_max + np.log(
            np.exp(gathered - row_max[:, None]).sum(axis=-1)
        )
    return out


def _forward_backward(emit: np.ndarray, model: _Model) -> tuple[np.ndarray, float]:
    """对数空间前向-后向，返回每状态后验 gamma (T,S) 与序列对数似然。"""

    t_total = emit.shape[0]
    n_states = model.n_states

    log_init = -math.log(n_states)  # 初始状态均匀（允许窗口首尾截断）
    alpha = np.empty((t_total, n_states), dtype=np.float64)
    alpha[0] = log_init + emit[0]

    for t in range(1, t_total):
        alpha[t] = emit[t] + _grouped_logsumexp(
            alpha[t - 1], None, model.pred_groups
        )

    last = alpha[t_total - 1]
    row_max = last.max()
    log_likelihood = row_max + np.log(np.exp(last - row_max).sum())
    if not math.isfinite(log_likelihood):
        raise RuntimeError("序列似然为零，参数与观测完全不相容，请检查输入")

    beta = np.zeros((t_total, n_states), dtype=np.float64)
    for t in range(t_total - 1, 0, -1):
        beta[t - 1] = _grouped_logsumexp(
            beta[t], emit[t], model.succ_groups
        )

    gamma = alpha + beta - log_likelihood
    gamma = np.clip(gamma, NEG_INF, 0.0)
    return gamma, log_likelihood


def infer_posterior(
    power: list[float | None],
    noise_std: float,
    device1: DeviceParams,
    device2: DeviceParams,
    switch_probability: float = 0.1,
) -> dict:
    """运行完整推断，返回两台设备逐分钟的开机后验概率与区间。

    参数
    ----
    power: 逐分钟总功率，元素为 None 表示该分钟缺测。
    noise_std: 电表读数噪声标准差（正值）。
    device1, device2: 两台设备的参数。
    switch_probability: 单台设备单分钟发生状态翻转的先验概率。

    返回
    ----
    字典，包含 posterior_on（2 x T 后验概率）、missing、log_likelihood 等。
    """

    if not power:
        raise ValueError("power 序列不能为空")
    t_total = len(power)
    if t_total > 1440:
        raise ValueError("观测序列最长为一天（1440 分钟）")

    raw = np.array(
        [np.nan if v is None else float(v) for v in power], dtype=np.float64
    )
    missing = np.isnan(raw)
    model = _build_model(device1, device2, switch_probability)
    emit = _log_emission(raw, missing, model, float(noise_std))
    gamma, log_likelihood = _forward_backward(emit, model)

    posterior = np.exp(gamma)  # (T, S)
    # 数值清理后重新归一化，保证逐时刻概率和为 1。
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    on1 = np.clip(posterior[:, model.state_s1 == 1].sum(axis=1), 0.0, 1.0)
    on2 = np.clip(posterior[:, model.state_s2 == 1].sum(axis=1), 0.0, 1.0)

    return {
        "posterior_on": [on1.tolist(), on2.tolist()],
        "missing": missing.tolist(),
        "log_likelihood": log_likelihood,
        "n_states": model.n_states,
        "window_minutes": t_total,
    }


def label_intervals(
    posterior_on: list[float],
    missing: list[bool],
    confidence_threshold: float,
) -> list[dict]:
    """把逐分钟后验概率压缩为 running / stopped / uncertain 区间。

    * p_on >= 阈值 -> running；p_on <= 1-阈值 -> stopped；其余 uncertain。
    * 相邻同标签分钟合并为半开区间 [start_minute, end_minute)。
    * 触及窗口首/尾的区间在对应边缘标记 truncated（状态被窗口截断）。
    * 区间置信度取区间内各分钟置信度的最小值（保守口径）。

    最短时长约束已在 HSMM 转移层强制，标签层不重复截断。
    """

    if not 0.5 < confidence_threshold < 1.0:
        raise ValueError("confidence_threshold 必须位于 (0.5, 1.0) 之间")

    n = len(posterior_on)
    labels: list[int] = []
    confidences: list[float] = []
    for p in posterior_on:
        if p >= confidence_threshold:
            labels.append(1)
            confidences.append(p)
        elif p <= 1.0 - confidence_threshold:
            labels.append(0)
            confidences.append(1.0 - p)
        else:
            labels.append(-1)
            confidences.append(max(p, 1.0 - p))

    intervals: list[dict] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            label = labels[start]
            state = {1: "running", 0: "stopped", -1: "uncertain"}[label]
            end = i
            truncated_edges: list[str] = []
            if start == 0:
                truncated_edges.append("start")
            if end == n:
                truncated_edges.append("end")
            missing_count = sum(1 for k in range(start, end) if missing[k])
            intervals.append(
                {
                    "state": state,
                    "start_minute": start,
                    "end_minute": end,
                    "duration_minutes": end - start,
                    "missing_minutes": missing_count,
                    "confidence": round(min(confidences[start:end]), 6),
                    "truncated_edges": truncated_edges,
                    "truncated": bool(truncated_edges),
                }
            )
            start = i
    return intervals
