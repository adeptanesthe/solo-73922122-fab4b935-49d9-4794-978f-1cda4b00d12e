"""请求/响应数据模型与输入校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_WINDOW = 1
MAX_WINDOW = 1440
MIN_DURATION = 1
MAX_DURATION = 30


class DeviceSpec(BaseModel):
    """单台设备的标定参数。"""

    name: str = Field(..., min_length=1, max_length=64, description="设备名称")
    on_power: float = Field(..., description="开机时该设备贡献的功率")
    off_power: float = Field(..., description="停机时该设备贡献的功率（待机功率）")
    min_on_minutes: int = Field(
        ..., ge=MIN_DURATION, le=MAX_DURATION, description="最短开机分钟数（1-30）"
    )
    min_off_minutes: int = Field(
        ..., ge=MIN_DURATION, le=MAX_DURATION, description="最短停机分钟数（1-30）"
    )

    @field_validator("on_power", "off_power")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("功率必须为有限数值")
        return v


class InferenceRequest(BaseModel):
    """推断接口请求体。"""

    power: list[float | None] = Field(
        ...,
        description=(
            "逐分钟总功率读数，长度 1-1440（至多一天）；"
            "null 表示该分钟缺测，缺测仍计入开停时长"
        ),
    )
    noise_std: float = Field(..., gt=0, description="读数噪声标准差（正值）")
    device1: DeviceSpec
    device2: DeviceSpec
    confidence_threshold: float = Field(
        0.95, gt=0.5, lt=1.0, description="置信度阈值，必须大于 0.5"
    )
    switch_probability: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="单台设备单分钟状态翻转的先验概率（0-1）",
    )

    @field_validator("power")
    @classmethod
    def _check_power_window(cls, v: list[float | None]) -> list[float | None]:
        if not (MIN_WINDOW <= len(v) <= MAX_WINDOW):
            raise ValueError("power 长度必须在 1 到 1440 分钟之间（至多一天）")
        for x in v:
            if x is not None and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("功率读数必须为有限数值或 null")
        return v

    @model_validator(mode="after")
    def _check_names(self) -> "InferenceRequest":
        if self.device1.name == self.device2.name:
            raise ValueError("两台设备的 name 不能相同")
        return self


class Interval(BaseModel):
    state: Literal["running", "stopped", "uncertain"]
    start_minute: int = Field(..., description="区间起始分钟（含，0 基）")
    end_minute: int = Field(..., description="区间结束分钟（不含，0 基）")
    duration_minutes: int
    missing_minutes: int = Field(..., description="区间内缺测分钟数")
    confidence: float = Field(
        ..., description="区间置信度（取区间内逐分钟置信度的最小值）"
    )
    truncated: bool = Field(..., description="区间是否在观测窗口首/尾被截断")
    truncated_edges: list[Literal["start", "end"]]


class DeviceResult(BaseModel):
    name: str
    posterior_on: list[float] = Field(
        ..., description="逐分钟处于开机状态的后验概率"
    )
    intervals: list[Interval]
    running_minutes: int
    stopped_minutes: int
    uncertain_minutes: int


class InferenceResponse(BaseModel):
    window_minutes: int
    missing_minutes: int
    confidence_threshold: float
    log_likelihood: float
    devices: list[DeviceResult]


class ScheduleRequest(BaseModel):
    """错峰开机排程请求体。

    读数窗口与双设备模型参数沿用 ``/infer``；新增待排程设备的功率、连续
    运行分钟数、候选开机分钟、线路功率上限与可接受的超限概率。
    """

    power: list[float | None] = Field(
        ...,
        description=(
            "逐分钟总功率读数，长度 1-1440（至多一天）；"
            "null 表示该分钟缺测，缺测仍计入开停时长"
        ),
    )
    noise_std: float = Field(..., gt=0, description="读数噪声标准差（正值）")
    device1: DeviceSpec
    device2: DeviceSpec
    switch_probability: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="单台设备单分钟状态翻转的先验概率（0-1）",
    )
    new_device_power: float = Field(
        ..., ge=0, description="待排程恒功率设备的功率（非负）"
    )
    run_minutes: int = Field(
        ..., ge=1, le=60, description="新设备连续运行分钟数（1-60）"
    )
    candidate_start_minutes: list[int] = Field(
        ...,
        min_length=1,
        description=(
            "候选开机分钟列表，1 表示读数窗口结束后的下一分钟；"
            "每个候选时段须完整落在接下来 60 分钟内"
        ),
    )
    line_power_limit: float = Field(
        ..., description="三台设备合计功率的线路功率上限"
    )
    max_exceedance_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="可接受的“任一分钟超限”概率上限",
    )

    @field_validator("power")
    @classmethod
    def _check_power_window(cls, v: list[float | None]) -> list[float | None]:
        if not (MIN_WINDOW <= len(v) <= MAX_WINDOW):
            raise ValueError("power 长度必须在 1 到 1440 分钟之间（至多一天）")
        for x in v:
            if x is not None and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("功率读数必须为有限数值或 null")
        return v

    @field_validator("new_device_power", "line_power_limit")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("功率必须为有限数值")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "ScheduleRequest":
        if self.device1.name == self.device2.name:
            raise ValueError("两台设备的 name 不能相同")
        last_allowed = 60 - self.run_minutes + 1
        for start in self.candidate_start_minutes:
            if not 1 <= start <= last_allowed:
                raise ValueError(
                    "候选时段必须完整落在读数窗口结束后的 60 分钟内："
                    f"开机分钟须位于 1 到 {last_allowed}"
                )
        return self


class ScheduleCandidate(BaseModel):
    start_minute: int = Field(..., description="开机分钟（1 = 窗口结束后的下一分钟）")
    end_minute: int = Field(..., description="运行结束分钟（含）")
    exceedance_probability: float = Field(
        ..., description="三台设备合计功率任一分钟超过上限的概率"
    )
    within_limit: bool = Field(..., description="是否不高于可接受的超限概率")


class ScheduleResponse(BaseModel):
    window_minutes: int = Field(..., description="读数窗口长度")
    horizon_minutes: int = Field(..., description="排程视野，固定为窗口后的 60 分钟")
    run_minutes: int
    new_device_power: float
    line_power_limit: float
    max_exceedance_probability: float
    found: bool = Field(..., description="是否存在满足概率上限的候选时段")
    start_minute: int | None = Field(
        None, description="最早可用开机分钟；无可用时段时为 null"
    )
    exceedance_probability: float | None = Field(
        None, description="对应最早可用时段的超限概率；无可用时段时为 null"
    )
    message: str
    candidates: list[ScheduleCandidate] = Field(
        ..., description="所有候选时段按开机分钟升序的评估结果"
    )

