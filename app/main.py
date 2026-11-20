"""食堂电表双设备状态推断 FastAPI 服务。"""

from __future__ import annotations

from fastapi import FastAPI

from .hsmm import DeviceParams, infer_posterior, label_intervals
from .scheduler import plan_schedule
from .schemas import (
    DeviceSpec,
    InferenceRequest,
    InferenceResponse,
    ScheduleRequest,
    ScheduleResponse,
)

app = FastAPI(
    title="双设备开停时段推断与错峰开机排程服务",
    description=(
        "根据一块电表的逐分钟总功率读数，用显式时长 HSMM 概率模型"
        "联合推断两台设备各自的开机/停机/待核实时段；并可据此为第三台"
        "恒功率设备选择线路功率超限风险可接受的最早开机分钟。"
    ),
    version="1.1.0",
)


def _to_params(spec: DeviceSpec) -> DeviceParams:
    return DeviceParams(
        on_power=spec.on_power,
        off_power=spec.off_power,
        min_on=spec.min_on_minutes,
        min_off=spec.min_off_minutes,
    )


@app.get("/health", tags=["运维"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/infer",
    response_model=InferenceResponse,
    tags=["推断"],
    summary="推断两台设备的开停时段",
)
def infer(req: InferenceRequest) -> InferenceResponse:
    device1 = _to_params(req.device1)
    device2 = _to_params(req.device2)

    result = infer_posterior(
        power=req.power,
        noise_std=req.noise_std,
        device1=device1,
        device2=device2,
        switch_probability=req.switch_probability,
    )

    specs = (req.device1, req.device2)
    missing = result["missing"]
    devices_out = []
    for idx in range(2):
        posterior = result["posterior_on"][idx]
        intervals = label_intervals(
            posterior_on=posterior,
            missing=missing,
            confidence_threshold=req.confidence_threshold,
        )
        running = sum(i["duration_minutes"] for i in intervals if i["state"] == "running")
        stopped = sum(i["duration_minutes"] for i in intervals if i["state"] == "stopped")
        uncertain = sum(i["duration_minutes"] for i in intervals if i["state"] == "uncertain")
        devices_out.append(
            {
                "name": specs[idx].name,
                "posterior_on": [round(p, 6) for p in posterior],
                "intervals": intervals,
                "running_minutes": running,
                "stopped_minutes": stopped,
                "uncertain_minutes": uncertain,
            }
        )

    return InferenceResponse(
        window_minutes=result["window_minutes"],
        missing_minutes=sum(1 for m in missing if m),
        confidence_threshold=req.confidence_threshold,
        log_likelihood=result["log_likelihood"],
        devices=devices_out,
    )


@app.post(
    "/schedule",
    response_model=ScheduleResponse,
    tags=["排程"],
    summary="为新设备选择超限风险可接受的最早开机分钟",
)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    result = plan_schedule(
        power=req.power,
        noise_std=req.noise_std,
        device1=_to_params(req.device1),
        device2=_to_params(req.device2),
        new_device_power=req.new_device_power,
        run_minutes=req.run_minutes,
        candidate_start_minutes=req.candidate_start_minutes,
        line_power_limit=req.line_power_limit,
        max_exceedance_probability=req.max_exceedance_probability,
        switch_probability=req.switch_probability,
    )
    return ScheduleResponse(**result)
