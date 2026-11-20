# 双设备开停时段推断服务

食堂值班员只有一块总电表，需要核对两台设备各自的开停时段。本服务接收
**至多一天（1–1440 分钟）的逐分钟总功率**，用显式时长隐半马尔可夫模型
（HSMM）联合推断两台设备的运行/停机/待核实时段。

## 模型与判定口径

- **隐状态**：两台设备状态的笛卡尔积 `(s1, s2)`，各自带“当前状态已持续
  分钟数”计数器；计数器到上限后封顶，任意长度的开/停段都能表达。
- **最短时长约束（1–30 分钟）**：设备只有在当前开/停状态持续满最短分钟数
  后才允许翻转，约束直接编码在状态转移表中。
- **观测模型**：总功率 = 两设备当前功率之和，叠加高斯噪声 `N(0, noise_std²)`；
  开机/停机功率分别标定，支持非零待机功率。
- **缺测分钟**：`null` 不产生观测似然，但转移照常推进，**缺测分钟仍计入
  开/停时长**，并在输出区间的 `missing_minutes` 中单独统计。
- **窗口首尾截断**：初始分布对所有增强状态均匀，不要求首个状态已持续满
  最短时长；触及窗口首/尾的区间用 `truncated` / `truncated_edges` 标记。
- **置信度判定**：由前向-后向算法得到逐分钟开机后验概率 `p`：
  - `p >= confidence_threshold` → `running`
  - `p <= 1 - confidence_threshold` → `stopped`
  - 其余 → `uncertain`（待核实）
  - 阈值必须 **大于 0.5**；区间置信度取区间内逐分钟置信度的最小值（保守口径）。
- 当两台设备功率组合本身不可区分（结构性歧义）时，相关分钟会自然落入
  `uncertain`，不会给出武断结论。

## 目录结构

```
app/
  hsmm.py      # HSMM 状态空间、转移表、前向-后向、区间标注
  schemas.py   # Pydantic 请求/响应模型与输入校验
  main.py      # FastAPI 路由（/health、/infer、/docs）
tests/         # 21 项 unittest 自测（模型性质 + 接口校验）
examples/      # 请求样例与调用脚本
Dockerfile
docker-compose.yml
```

## 用 Docker Compose 启动

```bash
docker compose up --build
```

服务监听 `http://127.0.0.1:8000`：

- 健康检查：`GET /health`
- 交互式文档：`GET /docs`（Swagger UI）
- OpenAPI Schema：`GET /openapi.json`
- 推断接口：`POST /infer`

停止：`docker compose down`

## 调用样例

### curl

```bash
curl -s http://127.0.0.1:8000/infer \
  -H 'Content-Type: application/json' \
  -d @examples/request.json | python -m json.tool
```

### Python

```bash
python examples/call_example.py
```

### 请求字段

| 字段 | 说明 |
| --- | --- |
| `power` | 逐分钟总功率数组，长度 1–1440；`null` 表示该分钟缺测 |
| `noise_std` | 读数噪声标准差，必须 > 0 |
| `device1` / `device2` | `name`、`on_power`、`off_power`、`min_on_minutes`、`min_off_minutes`（最短时长 1–30） |
| `confidence_threshold` | 置信度阈值，默认 0.95，必须 ∈ (0.5, 1.0) |
| `switch_probability` | 单设备单分钟翻转的先验概率，默认 0.1，∈ (0, 1) |

### 响应摘要

每台设备返回：

- `posterior_on`：逐分钟开机后验概率；
- `intervals`：合并后的区间，字段包括 `state`（running/stopped/uncertain）、
  `[start_minute, end_minute)` 半开区间、`duration_minutes`、
  `missing_minutes`、`confidence`、`truncated`、`truncated_edges`；
- `running_minutes` / `stopped_minutes` / `uncertain_minutes` 汇总。

## 本地开发与自测（不使用 Docker）

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

全天 1440 分钟、最短时长均为 30 的最坏规模（3600 个增强状态）在普通
CPU 上约 1 秒完成推断。
