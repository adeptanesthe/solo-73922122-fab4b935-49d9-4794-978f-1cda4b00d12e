"""食堂电表双设备状态推断 FastAPI 服务。"""

from __future__ import annotations

from fastapi import FastAPI

from .hsmm import DeviceParams, infer_posterior, label_intervals
from .schemas import InferenceRequest, InferenceResponse

app = FastAPI(
    title="双设备开停时段推断服务",
    description=(
        "根据一块电表的逐分钟总功率读数，用显式时长 HSMM 概率模型"
        "联合推断两台设备各自的开机/停机/待核实时段。"
    ),
    version="1.0.0",
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
    device1 = DeviceParams(
        on_power=req.device1.on_power,
        off_power=req.device1.off_power,
        min_on=req.device1.min_on_minutes,
        min_off=req.device1.min_off_minutes,
    )
    device2 = DeviceParams(
        on_power=req.device2.on_power,
        off_power=req.device2.off_power,
        min_on=req.device2.min_on_minutes,
        min_off=req.device2.min_off_minutes,
    )

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
