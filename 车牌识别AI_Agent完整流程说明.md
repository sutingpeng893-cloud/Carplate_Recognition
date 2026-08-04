# 语音车牌识别 AI Agent 完整业务流转说明

> 文档版本：2026-07-31 | 基于 `realtime_audio_demo/services/plate_agent*.py` 全模块

---

## 一、整体架构概览

系统分为**用户侧**与**Agent侧**两部分，以一轮用户语音输入为单位进行处理。每轮音频进入后，Agent独立完成识别→校验→确认→纠错的完整子流程，最终向用户返回话术。

```
用户侧                                Agent侧
  │  ─── 音频(WAV字节流) ───────────►  PlateAgentService.handle_audio_turn()
  │                                      │
  │  ◄── ACK衔接语(0/1/3/5秒) ─────────  emit_scheduled_acks()
  │                                      │
  │                                    Agent循环（最多8次迭代）
  │                                      │  plan_next_action() → 大模型
  │                                      │  execute_all() → 工具执行
  │                                      │  observations → 回填上下文
  │                                      │
  │  ◄── 最终响应(JSON+话术) ──────────  build_final_result()
  │
  └── 用户确认/纠错 → 下一轮音频输入
```

---

## 二、核心状态对象 `PlateAgentState`

位于 `plate_agent_types.py`，是整个 Agent 的核心状态容器。

| 字段 | 类型 | 说明 |
|------|------|------|
| `car_plate` | `str` | 当前暂存车牌（每次编辑后实时更新） |
| `plate_chars` | `list[PlateCharState]` | 车牌每个字符的详细状态 |
| `confirmed` | `bool` | 整车是否已全部确认 |
| `need_confirm_chars` | `list[PlateCharState]` | 仍需用户二次确认的字符（易混淆且未确认） |
| `confirmed_chars` | `list[PlateCharState]` | 已确认字符（用户明确确认或本轮新输入） |
| `vehicle_type` | `str` | `fuel`/`new_energy`/`unknown` |
| `confusions` | `list[PlateConfusion]` | 规则检测出的易混淆字符原始列表 |
| `final_car_plate` | `str` | 最终确认车牌（仅 `is_confirmed=True` 时有值） |
| `ack_sent` | `bool` | 本轮 ACK 是否已发送 |
| `turn_summaries` | `list[str]` | 历史轮次摘要（最近6轮，注入下轮状态栏） |

**关键属性：**
- `has_car_plate`：`car_plate` 或 `plate_chars` 非空则为 True
- `is_confirmed`：`confirmed` 或 `final_car_plate` 非空则为 True

---

## 三、主流程详解

### 3.1 入口：`handle_audio_turn()`

**位置：** `plate_agent.py → PlateAgentService.handle_audio_turn()`

```python
async def handle_audio_turn(
    *, model, wav_bytes, state, session_id, on_ack,
    turn_summaries, prior_agent_history, include_confirmation_reply
) -> PlateAgentResult
```

**执行步骤：**

1. **克隆状态**：`clone_state(state)` — 防止副作用污染外部状态
2. **同步历史摘要**：将 `turn_summaries` 注入 `working.turn_summaries`（最近6轮）
3. **发送 ACK**：`emit_compat_ack_if_needed()` — 立即发第一条衔接语
4. **启动 Agent 循环**：最多 `MAX_AGENT_ITERATIONS=8` 次迭代
5. **生成最终结果**：`build_final_result()`

---

### 3.2 ACK 即时反馈机制

**位置：** `plate_agent_ack.py`

在大模型处理期间，系统按时间点向前端发送衔接语，避免用户等待无反馈：

```
ack_schedule_for_state(state) → ack_scene_for_state() → 判断场景
  - has_car_plate = False → 场景 "initial"（首轮）
  - has_car_plate = True  → 场景 "update"（多轮）
```

| 时间点 | 首轮（initial）内容 | 多轮（update）内容 |
|--------|-------------------|------------------|
| 0秒 | "语音已收到，正在判断是否包含车牌信息。" | "语音已收到，正在判断您是在确认还是修改。" |
| 1秒 | "正在识别车牌号码内容。" | "正在结合当前车牌处理您的这次回复。" |
| 3秒 | "还在结合车牌规则和发音做确认。" | "还在复核修改结果和需要确认的位置。" |
| 5秒 | "识别还在处理，请稍等。" | "处理还在继续，请稍等。" |

只有当 `is_result_ready()` 仍为 False 时才发出，结果就绪则停止。

---

### 3.3 Agent 规划循环

**位置：** `plate_agent.py → handle_audio_turn() 主循环`

每次迭代调用 `plan_next_action()`：

- **首轮（iteration=1）**：传入音频 + 状态栏（`build_plate_agent_status_bar()`）
- **续轮（iteration>1）**：传入 observations 文本，不含音频

模型输出两类结构：

```json
// 需要执行工具
{"thought": "当前没有车牌，识别音频后需要写入", "tool_calls": [{"name": "set_plate", "arguments": {"car_plate": "津A12345"}}]}

// 结束本轮
{"thought": "用户确认整车", "finish": {"task_status": "confirmed", "reply_scene": "confirmed"}}
```

解析函数：`parse_agent_plan(raw_output)` → 返回 `PlateAgentPlan(thought, tool_calls, finish)`

---

### 3.4 工具执行层 `PlateToolExecutor`

**位置：** `plate_agent_tooling.py`

所有真实状态修改都经由此类执行。每次工具调用返回：

```python
{
    "tool_call_id": "call_1",
    "name": "set_plate",
    "success": True,
    "message": "车牌状态已设置...",
    "data": {...},
    "before_state": {...},  # 执行前状态快照
    "after_state": {...}    # 执行后状态快照
}
```

---

## 四、车牌识别与校验逻辑（上级重点关注）

### 4.1 首轮识别流程

**分支条件：** `working.has_car_plate == False`

1. 大模型从音频中识别车牌文本
2. 调用 `set_plate(car_plate)` 工具
3. 工具内部执行：

```python
# plate_agent_tooling.py → _set_plate()
plate = normalize_candidate_plate(args["car_plate"])
if not is_valid_plate_number(plate):
    return False, "车牌格式不合法，状态未更新", {...}

confirmed_positions = parse_positions(args.get("confirmed_positions"), len(plate))
self._write_plate(plate, confirmed_positions=confirmed_positions)
```

4. `_write_plate()` 内部：
   - 调用 `refresh_plate_state()` 全量刷新
   - 调用 `_refresh_confirmation_after_plate_write()` 按规则初始化待确认列表

### 4.2 车牌有效性校验 `is_valid_plate_number()`

**位置：** `plate_agent_rules.py`

```python
def is_valid_plate_number(car_plate: str) -> bool:
    plate = normalize_plate_text(car_plate)
    if vehicle_type_by_length(plate) == "unknown":  # 非7位/8位 → 无效
        return False
    if plate[0] not in PROVINCE_ABBREVIATIONS:       # 首位非省份简称 → 无效
        return False
    if not re.fullmatch(r"[A-Z]", plate[1]):         # 第2位非字母 → 无效
        return False
    for index, char in enumerate(plate[2:], start=3):
        is_tail = (index == len(plate))
        if char in SPECIAL_PLATE_TAIL_CHARS:         # 尾字：警临学领挂，只能在最后一位
            if not is_tail: return False
            continue
        if not re.fullmatch(r"[A-Z0-9]", char):     # 其余位：数字或大写字母
            return False
    return True
```

**车牌长度类型：**

```python
def vehicle_type_by_length(car_plate: str) -> str:
    length = len(car_plate)
    if length == 7: return "fuel"         # 普通燃油车
    if length == 8: return "new_energy"   # 新能源
    return "unknown"                      # 其他长度无效
```

**特殊归一化：**
- `normalize_plate_text()` 会自动将 `G` 开头转为 `冀`（`replace_leading_g_with_ji()`）

### 4.3 易混淆字符检测规则 `detect_initial_confusions_by_rule()`

**位置：** `plate_agent_rules.py`

触发二次确认的两类字符：

| 类别 | 字符集 | 适用位置 |
|------|--------|--------|
| 省份易混 | `甘 赣 津 京 桂 贵 冀 吉` | 第1位（省份位） |
| 字母数字易混 | `2 R 1 E` | 任意位 |

```python
def detect_initial_confusions_by_rule(car_plate: str) -> list[PlateConfusion]:
    confusions = []
    if plate[0] in CONFUSION_PROVINCE_CHARS:          # 省份首位易混
        confusions.append(build_confusion(position=1, value=plate[0]))
    for index, value in enumerate(plate, start=1):
        if value in CONFUSION_ALNUM_CHARS:            # 字母数字易混
            confusions.append(build_confusion(position=index, value=value))
    return with_relative_confusion_reasons(plate, confusions)
```

每个 `PlateConfusion` 包含：
- `position`：从1开始的位置
- `value`：当前识别的字符
- `reason`：口语说明（"第1位当前识别为天津的津，请用户确认是否正确。"）
- `candidates`：候选字符列表

### 4.4 字符描述函数 `describe_plate_char()`

将字符转换为用户可听懂的口语描述：

| 字符 | 返回描述 |
|------|---------|
| `赣` | "江西的赣" |
| `甘` | "甘肃的甘" |
| `津` | "天津的津" |
| `京` | "北京的京" |
| `桂` | "广西的桂" |
| `贵` | "贵州的贵" |
| `冀` | "河北的冀" |
| `吉` | "吉林的吉" |
| `2` | "数字 2" |
| `R` | "字母 R" |
| `1` | "数字 1" |
| `E` | "字母 E" |

---

## 五、车牌修改逻辑（上级重点关注）

### 5.1 三类编辑工具

#### 5.1.1 `replace_position(position, value)` — 替换指定位

```python
# plate_agent_edit.py → apply_plate_edit_command()
if action == "replace_position":
    if not valid_existing_position(command.position, chars) or not command.value:
        return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
    chars[command.position - 1] = command.value
    return edit_success(chars, command, changed_positions=[command.position])
```

**状态更新后：** 被替换位自动加入 `confirmed_chars`，其他位确认状态保留。

#### 5.1.2 `insert_position(position, value, relation)` — 插入字符

```python
if action == "insert_position":
    insert_index = command.position if command.relation == "after" else command.position - 1
    chars.insert(insert_index, command.value)
    return edit_success(chars, command, changed_positions=[insert_index + 1])
```

**状态更新后：** 插入位自动加入 `confirmed_chars`，插入点后的位置编号后移1。

#### 5.1.3 `delete_position(position)` — 删除指定位

```python
if action == "delete_position":
    if not valid_existing_position(command.position, chars):
        return edit_error(plate, command, EDIT_UNCLEAR_REPLY)
    del chars[command.position - 1]
    return edit_success(chars, command, changed_positions=[])
```

**状态更新后：** 删除位当前新字符重新计算易混淆，其他位确认状态随位置移动保留。

#### 5.1.4 `replace_char(old_value, value)` — 按字符值替换

支持 `occurrence`：`first`/`last`/`all`，当车牌中该字符出现多次但未指定时返回错误：`EDIT_DUPLICATE_CHAR_TEMPLATE`。

### 5.2 编辑后确认状态保留逻辑

**位置：** `plate_agent_tooling.py → _preserved_confirmation_positions_after_edit()`

编辑操作后，通过位置映射保留已有确认状态：

```
替换操作：
  - 被替换位：从 pending/confirmed 列表移除
  - 其他位：位置不变，保留原确认状态
  - 被替换位：自动加入 confirmed_chars

插入操作：
  - 插入点之前的位：保留原确认状态（位置不变）
  - 插入点之后的位：位置 +1，保留原确认状态
  - 插入位：自动加入 confirmed_chars

删除操作：
  - 被删除位：从 pending/confirmed 移除
  - 被删除位之后的位：位置 -1，保留原确认状态
  - 删除后当前该位：重新按规则判断是否需要二次确认
```

### 5.3 编辑有效性校验

编辑工具执行后，若结果不合法：
```python
if not is_valid_plate_number(result.car_plate):
    return False, "编辑后的车牌格式不合法，状态未更新", public_edit_result(result)
```
**状态不会更新**，返回失败，话术使用 `EDIT_INVALID_REPLY`。

### 5.4 语音字符规范化

**位置：** `plate_agent_edit.py → SPOKEN_PLATE_CHAR_REPLACEMENTS`

语音口语表达会自动转换为标准字符：

| 口语 | 字符 |
|------|------|
| 零、〇、洞 | 0 |
| 幺、么、一 | 1 |
| 二、两 | 2 |
| 是 | 4 |
| 陆 | 6 |
| 拐 | 7 |
| 吸 | C |
| 勾、沟儿 | J |
| 圈 | Q |

---

## 六、确认状态管理

### 6.1 确认动作

**位置：** `plate_agent_confirmation.py → apply_confirmation_actions()`

| 动作 | 效果 |
|------|------|
| `add_need_confirmation` | 添加到 `need_confirm_chars`（仅易混淆位生效） |
| `remove_need_confirmation` | 从 `need_confirm_chars` 移除 |
| `clear_need_confirmation` | 清空 `need_confirm_chars` |
| `add_confirmed` | 加入 `confirmed_chars`，从 `need_confirm_chars` 移除 |
| `remove_confirmed` | 从 `confirmed_chars` 移除 |
| `confirm_all` | 清空 `need_confirm_chars`，所有位加入 `confirmed_chars` |

**关键限制：** `add_need_confirmation` 只对 `is_rule_confusion_position()` 返回 True 的位置生效，防止 Agent 随意添加非规则定义的待确认位。

### 6.2 整车确认条件

**位置：** `plate_agent.py → can_confirm_from_finish()`

```python
def can_confirm_from_finish(working, observations) -> bool:
    if working.final_car_plate:                     # 已有最终车牌
        return True
    if not is_valid_plate_number(working.car_plate): # 车牌不合法
        return False
    return not turn_wrote_plate(observations)        # 本轮未执行过写入/编辑工具
```

**整车确认后执行：** `confirm_current_plate_from_finish()` → `apply_confirmation_actions(confirm_all)` + 设置 `final_car_plate`

---

## 七、状态流转完整路径

### 路径一：首轮识别成功

```
接收音频
  → Agent: set_plate("津A12345")
  → refresh_plate_state() 刷新
  → detect_initial_confusions_by_rule() → 发现 津(第1位)、1(第3位) 需确认
  → need_confirm_chars: [第1位=津, 第3位=1]
  → finish: {task_status: "need_confirmation", reply_scene: "initial_success"}
  → 输出话术: "我识别到的车牌号是津A12345。需要您确认：第1位是天津的津，第3位是数字1。"
```

### 路径二：多轮纠错

```
当前状态: car_plate="津A12345", need_confirm_chars=[第1位=津, 第3位=1]
用户语音: "第3位是字母E"
  → Agent: replace_position(position=3, value="E")
  → apply_plate_edit_command() → 新车牌 "津AE2345"? 不对，应是 "津A12345" 第3位替换为E → "津AE2345"
  → is_valid_plate_number("津AE2345") → True
  → _preserved_confirmation_positions_after_edit():
      - 第3位被替换 → 从 need_confirm_chars 移除，加入 confirmed_chars
      - 第1位保留在 need_confirm_chars
  → finish: {task_status: "need_confirmation", reply_scene: "update_success"}
  → 输出话术: "已按您的说明更新为津AE2345。需要您确认：第1位是天津的津。"
```

### 路径三：用户整车确认

```
当前状态: car_plate="津AE2345", need_confirm_chars=[]
用户语音: "对的，就是这个"
  → Agent: 没有 tool_calls
  → finish: {task_status: "confirmed", reply_scene: "confirmed"}
  → can_confirm_from_finish(): turn_wrote_plate()=False → 可以确认
  → confirm_current_plate_from_finish()
  → final_car_plate = "津AE2345"
  → 输出话术: "好的，已确认您的车牌号是津AE2345。"
```

---

## 八、最终响应生成 `build_final_result()`

**位置：** `plate_agent.py → PlateAgentService.build_final_result()`

根据 `finish_status` 和状态决策输出话术：

| 条件 | task_status | 话术来源 |
|------|-------------|---------|
| confirmed + 有 final_car_plate | `confirmed` | `build_confirmed_reply()` |
| 无车牌 + invalid | `invalid` | `INVALID_PLATE_REPLY` |
| 无车牌 | `need_more_info` | `NO_PLATE_REPLY` |
| invalid + 有失败记录 | `need_confirmation` | `build_edit_invalid_reply()` |
| 状态未变化 + 无法解析 | `need_confirmation` | `reply_for_failed_or_unclear_edit()` |
| 其他 | `need_confirmation` | `reply_for_current_state()` |

**reply_scene 到模板的映射：**

| reply_scene | 调用 |
|-------------|------|
| `initial_success` | `build_initial_success_reply()` |
| `update_success` | `build_update_success_reply()` |
| `partial_confirmation` | `build_partial_confirmation_reply()` |
| `confirmed` | `build_confirmed_reply()` |

---

## 九、历史摘要机制 `plate_agent_history.py`

每轮结束后生成摘要，注入下轮状态栏（最近6轮）：

```python
def build_plate_turn_summary(*, turn_index, before_state, result) -> str:
    # 记录内容：轮次、车牌变化、已确认字符、待确认字符、AI回复
    # 字数限制：900字符（摘要），180字符（AI回复片段）
```

摘要被注入到 `agent_status` 中的"最近历史"部分，帮助模型理解多轮上下文。

---

## 十、兼容降级机制

部分 Qwen/OpenAI 兼容服务不支持 `role=tool` 的消息格式，系统提供降级处理：

- **方法：** `compatible_agent_history()` — 把 `tool` role 消息转为 `user` 文本消息
- **触发：** 首次请求返回状态码错误时，自动重试降级版本
- **不影响功能：** 仅影响消息格式，状态管理和逻辑不变
