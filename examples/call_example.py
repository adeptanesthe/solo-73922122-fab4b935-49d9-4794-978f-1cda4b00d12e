"""调用示例：向本地推断服务发送请求并打印两台设备的时段摘要。

运行前提：服务已通过 docker compose up 启动（或本地 uvicorn 运行在 8000 端口）。
用法：  python examples/call_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/infer"
STATE_CN = {"running": "运行", "stopped": "停机", "uncertain": "待核实"}


def main() -> None:
    payload = json.loads(
        (Path(__file__).parent / "request.json").read_text(encoding="utf-8")
    )
    resp = httpx.post(URL, json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    print(f"窗口长度: {data['window_minutes']} 分钟，缺测: {data['missing_minutes']} 分钟")
    print(f"序列对数似然: {data['log_likelihood']:.2f}")
    print(f"置信度阈值: {data['confidence_threshold']}")
    for dev in data["devices"]:
        print(f"\n== {dev['name']} ==")
        print(
            f"运行 {dev['running_minutes']} 分钟 / "
            f"停机 {dev['stopped_minutes']} 分钟 / "
            f"待核实 {dev['uncertain_minutes']} 分钟"
        )
        for iv in dev["intervals"]:
            edge = f" [窗口边缘截断: {','.join(iv['truncated_edges'])}]" if iv["truncated"] else ""
            miss = f"，含缺测 {iv['missing_minutes']} 分钟" if iv["missing_minutes"] else ""
            print(
                f"  {STATE_CN[iv['state']]:>3} "
                f"[{iv['start_minute']:>3}, {iv['end_minute']:>3}) "
                f"{iv['duration_minutes']:>2} 分钟，"
                f"置信度 {iv['confidence']:.3f}{miss}{edge}"
            )


if __name__ == "__main__":
    main()
