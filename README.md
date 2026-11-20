# 双设备开停时段推断与错峰开机排程服务

食堂值班员只有一块总电表，需要核对两台设备各自的开停时段。本服务接收
**至多一天（1–1440 分钟）的逐分钟总功率**，用显式时长隐半马尔可夫模型
（HSMM）联合推断两台设备的运行/停机/待核实时段；并可在同一模型上为第三台
**恒功率设备**选择线路功率超限风险可接受的最早开机分钟（`/schedule`）。

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

## 错峰开机排程（`/schedule`）

沿用 `/infer` 的读数窗口与双设备模型参数，为第三台恒功率设备挑选开机分钟：

- **起点**：以读数窗口**最后一分钟的增强状态联合后验** `(s1, s2, c1, c2)`
  为初始分布，两台设备各自当前开/停状态已持续的分钟数原样保留，因此
  “还要多久才允许翻转”不会被抹掉；两台设备之间也不做独立性近似。
- **时间轴**：窗口结束后的下一分钟记为**第 1 分钟**，视野固定为接下来
  **60 分钟**。候选时段 `[start, start + run_minutes - 1]` 必须完整落在
  视野内（`start + run_minutes - 1 <= 60`），越界请求返回 422。
- **未来演化**：不再有观测，只按原转移表（含最短开/停时长与翻转先验）滚动。
  新设备从候选分钟起以恒定功率连续运行 `run_minutes` 分钟，其余分钟不计功率。
- **超限判定**：每分钟用完整增强状态判定“两台已知设备 + 新设备”合计功率是否
  `> line_power_limit`。超限的状态质量从存活分布中剔除且不再回流，
  因此**同一条未来路径多次超限只计一次**；最终
  `P(任一分钟超限) = 1 - 存活质量`。
- **选点**：对所有候选计算上述概率，返回**最早**一个不超过
  `max_exceedance_probability` 的开机分钟及其概率；全部不合格时返回
  `found=false`、`start_minute=null`、`message="无可用时段"`。
- **数值口径**：概率为对未来路径并集的无条件概率；响应中的
  `exceedance_probability` 保留 6 位小数，而 `within_limit` 用未取整值判定，
  避免四舍五入把轻微超限变成合格。

## 目录结构

```
app/
  hsmm.py      # HSMM 状态空间、转移表、前向-后向、区间标注
  scheduler.py # 末分钟增强状态后验滚动、超限并集概率、最早可行开机分钟
  schemas.py   # Pydantic 请求/响应模型与输入校验
  main.py      # FastAPI 路由（/health、/infer、/schedule、/docs）
tests/         # 40 项 unittest 自测（模型性质 + 排程口径 + 接口校验）
examples/      # 推断/排程请求样例与调用脚本
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
- 排程接口：`POST /schedule`

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

## 排程请求样例

### curl

```bash
curl -s http://127.0.0.1:8000/schedule \
  -H 'Content-Type: application/json' \
  -d @examples/schedule_request.json | python -m json.tool
```

### Python

```bash
python examples/call_schedule_example.py
```

### 排程请求字段

| 字段 | 说明 |
| --- | --- |
| `power` / `noise_std` / `device1` / `device2` / `switch_probability` | 与 `/infer` 完全相同的读数窗口与双设备模型参数 |
| `new_device_power` | 待排程恒功率设备功率，非负 |
| `run_minutes` | 新设备连续运行分钟数，1–60 |
| `candidate_start_minutes` | 候选开机分钟列表（1 起），每个时段须完整落在 60 分钟视野内 |
| `line_power_limit` | 三台设备合计功率的线路功率上限 |
| `max_exceedance_probability` | 可接受的“任一分钟超限”概率上限，0–1 |

### 排程响应摘要

- `found` / `start_minute` / `exceedance_probability` / `message`：最早可行
  开机分钟及其超限概率；无可用时段时 `found=false`、前两者为 `null`、
  `message="无可用时段"`。
- `candidates`：所有候选按开机分钟升序，每项含 `start_minute`、
  `end_minute`、`exceedance_probability`、`within_limit`。
- `window_minutes` / `horizon_minutes` / `run_minutes` 等：回显本次计算口径。

样例请求的读数窗口为 60 分钟，前 57 分钟两台设备均停机、最后 3 分钟抽风机
（30 W）刚开机；新设备 15 W 运行 10 分钟、线路上限 40 W、可接受超限概率 0.7。
实测返回 `found=true`、`start_minute=31`、`exceedance_probability=0.690185`，
第 1/11/21 分钟的超限概率依次为 0.965083 / 0.767545 / 0.708005（均不合格）。

## 本地开发与自测（不使用 Docker）

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

全天 1440 分钟、最短时长均为 30 的最坏规模（3600 个增强状态）在普通
CPU 上约 1 秒完成推断。
