"""调用示例：为第三台恒功率设备挑选最早的安全开机分钟。

运行前提：服务已通过 docker compose up 启动（或本地 uvicorn 运行在 8000 端口）。
用法：  python examples/call_schedule_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/schedule"


def main() -> None:
    payload = json.loads(
        (Path(__file__).parent / "schedule_request.json").read_text(encoding="utf-8")
    )
    resp = httpx.post(URL, json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    print(
        f"读数窗口 {data['window_minutes']} 分钟；排程视野 {data['horizon_minutes']} 分钟；"
        f"新设备 {data['new_device_power']} W 连续运行 {data['run_minutes']} 分钟"
    )
    print(
        f"线路功率上限 {data['line_power_limit']} W；"
        f"可接受超限概率 <= {data['max_exceedance_probability']}"
    )
    print("\n候选开机分钟的超限概率（同一条路径多次超限只计一次）：")
    for c in data["candidates"]:
        mark = "可用" if c["within_limit"] else "超限"
        print(
            f"  第 {c['start_minute']:>2}-{c['end_minute']:>2} 分钟  "
            f"P(超限)={c['exceedance_probability']:.6f}  [{mark}]"
        )
    print()
    if data["found"]:
        print(
            f"结论：最早可用开机分钟为第 {data['start_minute']} 分钟"
            f"（{data['start_minute']}-{data['start_minute'] + data['run_minutes'] - 1} 分钟），"
            f"超限概率 {data['exceedance_probability']:.6f}"
        )
    else:
        print(f"结论：{data['message']}")


if __name__ == "__main__":
    main()
