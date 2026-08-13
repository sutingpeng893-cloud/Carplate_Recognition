# 车牌识别多轮流程优化方案（Pending Action 模式）

**目标：** 每轮交互最多调用 2 次 LLM，平均约 1.5 次

**核心改动：** 引入 `pending_plate` 状态，大模型提取修改意图后先追问用户确认，用户确认后才写入状态；移除 `refresh_confusions_after_audio` 调用

---

## 新增状态字段

```python
@dataclass(slots=True)
class PlateAgentState:
    # ... 现有字段 ...
    pending_plate: str = ""                                   # 预执行后的新车牌，待用户确认
    pending_commands: list[PlateEditCommand] = field(default_factory=list)  # 对应的编辑指令
```

---

## 新增话术模板

```python
# plate_agent_messages.py

PENDING_ACTION_CONFIRM_TEMPLATE = "当前暂存车牌是{old_plate}，识别到您要{action_desc}，修改后车牌{new_plate}，是否执行？"
PENDING_ACTION_APPLIED_TEMPLATE = "已修改为{plate}，请确认是否正确。"
PENDING_ACTION_DISCARDED_TEMPLATE = "好的，当前保留原车牌{plate}，请继续确认或告知修改。"
```

需要实现 `describe_edit_commands(commands: list[PlateEditCommand]) -> str`：
```python
def describe_edit_command(cmd: PlateEditCommand) -> str:
    if cmd.action == "replace_position":
        return f"将第{cmd.position}位替换为{cmd.value}"
    if cmd.action == "replace_char":
        return f"将{cmd.old_value}替换为{cmd.value}"
    if cmd.action == "insert_position":
        relation = "之后" if cmd.relation == "after" else "之前"
        return f"在第{cmd.position}位{relation}插入{cmd.value}"
    if cmd.action == "delete_position":
        return f"删除第{cmd.position}位"
    return ""
```

---

## 多轮入口分流

**入口条件：** `working.has_car_plate = True`（首轮识别已成功）

**状态检查点：** `working.pending_plate` 是否为空

```
├─ pending_plate 为空     → 【分支 A：普通修改轮】
└─ pending_plate 不为空   → 【分支 B：等待确认待执行修改轮】
```

---

## 【分支 A：普通修改轮】 `pending_plate=""`

### A1. 检测用户意图：确认整车牌 or 给修改意见

**LLM 调用 1：** `detect_confirmation(model, wav_bytes, state)`
- **Prompt 职责：** 判断用户是否明确确认当前完整车牌
- **输入上下文：** `car_plate`、`assistant_reply`、`turn_summaries`
- **返回：** `bool` (True=确认整车牌 / False=不是确认)

---

#### A1-分支1：用户确认整车牌 → **终止流程**

**条件：** `confirmation == True`

**状态变更：**
```python
working.final_car_plate = working.car_plate
working.confirmed = True
working.need_confirm_chars = []
working.confirmed_chars = [全部字符标记为已确认]
```

**回复话术：**
```python
assistant_reply = build_confirmed_reply(working.final_car_plate)
# 例："好的，已确认您的车牌号是京AB123。"
```

**输出JSON：**
```json
{
  "task_status": "confirmed",
  "car_plate": "京AB123",
  "final_car_plate": "京AB123",
  "assistant_reply": "好的，已确认您的车牌号是京AB123。"
}
```

**LLM 调用次数：** 1 次  
**流程结束**

---

#### A1-分支2：用户给出修改意见 → **继续 A2**

**条件：** `confirmation == False`

---

### A2. 提取编辑 action 并预执行

**LLM 调用 2：** `update_car_plate(model, wav_bytes, state)`
- **Prompt 职责：** 从音频中提取编辑动作（replace / insert / delete / none）
- **输入上下文：** `car_plate`、`edit_steps=[]`
- **返回：** `PlateEditResult` 包含：
  - `commands: list[PlateEditCommand]`
  - `car_plate: str` - **预执行后的新车牌**（尚未写入状态）
  - `changed: bool`
  - `error: str`

---

#### A2-分支1：未识别到有效修改动作

**条件：** `edit_result.changed == False` 且 `command.action in ("none", "unknown")`

**状态变更：** 无（保持原车牌）

**回复话术：**
```python
assistant_reply = edit_result.error or EDIT_UNCLEAR_REPLY
# 例："我没有听清您要修改车牌的哪一处，当前仍保留原来的车牌。请您说明要替换、插入或删除哪一位。"
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "assistant_reply": "我没有听清您要修改车牌的哪一处..."
}
```

**LLM 调用次数：** 2 次  
**下一轮：** pending_plate 仍为空 → 再次进入分支 A

---

#### A2-分支2：识别到修改动作，预执行后检验格式

**条件：** `edit_result.changed == True`

**格式校验：** `is_valid_plate_number(edit_result.car_plate)`

##### A2-分支2-1：格式不合规

**条件：** `not is_valid_plate_number(new_plate)`

**状态变更：** 无（不保存预执行结果）

**回复话术：**
```python
assistant_reply = EDIT_INVALID_REPLY.format(plate=working.car_plate)
# 例："按这次修改后车牌格式不符合规则，当前仍保留车牌京AB123。请您重新说明要改哪一处。"
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "assistant_reply": "按这次修改后车牌格式不符合规则..."
}
```

**LLM 调用次数：** 2 次  
**下一轮：** pending_plate 仍为空 → 再次进入分支 A

---

##### A2-分支2-2：格式合规，保存为待确认动作

**条件：** `is_valid_plate_number(new_plate) == True`

**状态变更：**
```python
working.pending_plate = edit_result.car_plate
working.pending_commands = [cmd.to_dict() for cmd in commands]  # 序列化保存
# 注意：working.car_plate 保持不变（尚未应用修改）
```

**回复话术：**
```python
action_desc = describe_edit_commands(commands)
assistant_reply = PENDING_ACTION_CONFIRM_TEMPLATE.format(
    old_plate=working.car_plate,
    action_desc=action_desc,
    new_plate=working.pending_plate
)
# 例："当前暂存车牌是京AB123，识别到您要将第3位替换为E，修改后车牌京ABE23，是否执行？"
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "pending_plate": "京ABE23",
  "assistant_reply": "当前暂存车牌是京AB123，识别到您要将第3位替换为E，修改后车牌京ABE23，是否执行？"
}
```

**LLM 调用次数：** 2 次  
**下一轮：** pending_plate 不为空 → 进入【分支 B】

---

## 【分支 B：等待确认待执行修改轮】 `pending_plate!=""`

**入口条件：** 上一轮已保存 `pending_plate` 和 `pending_commands`

### B1. 单次 LLM 调用：分类用户回复

**LLM 调用 1（新实现）：** `classify_pending_response(model, wav_bytes, state)`
- **Prompt 职责：**
  - 判断用户对"是否执行X修改"的回复类型
  - 若用户给了新修改意见，同时提取新的编辑 action
- **输入上下文：**
  - `car_plate`（当前车牌）
  - `pending_plate`（待执行的新车牌）
  - `pending_commands`（对应的修改动作描述）
  - `assistant_reply`（上轮回复："是否执行X"）
- **返回：** JSON 结构
  ```json
  {
    "intent": "execute" | "reject" | "new_edit",
    "commands": [...] // 仅当 intent="new_edit" 时有值
  }
  ```

---

#### B1-分支1：用户确认执行

**条件：** `response.intent == "execute"`

**状态变更：**
```python
working.car_plate = working.pending_plate
working.pending_plate = ""
working.pending_commands = []
refresh_plate_state(working, working.car_plate, ...)
```

**回复话术：**
```python
assistant_reply = PENDING_ACTION_APPLIED_TEMPLATE.format(plate=working.car_plate)
# 例："已修改为京ABE23，请确认是否正确。"
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京ABE23",
  "assistant_reply": "已修改为京ABE23，请确认是否正确。"
}
```

**LLM 调用次数：** 1 次  
**下一轮：** pending_plate 为空 → 回到【分支 A】

---

#### B1-分支2：用户拒绝执行，无新修改意见

**条件：** `response.intent == "reject"`

**状态变更：**
```python
working.pending_plate = ""
working.pending_commands = []
# working.car_plate 保持不变
```

**回复话术：**
```python
assistant_reply = PENDING_ACTION_DISCARDED_TEMPLATE.format(plate=working.car_plate)
# 例："好的，当前保留原车牌京AB123，请继续确认或告知修改。"
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "assistant_reply": "好的，当前保留原车牌京AB123，请继续确认或告知修改。"
}
```

**LLM 调用次数：** 1 次  
**下一轮：** pending_plate 为空 → 回到【分支 A】

---

#### B1-分支3：用户拒绝执行 + 给出新修改意见

**条件：** `response.intent == "new_edit"` 且 `response.commands` 非空

**处理流程：** 与【分支 A2】相同

1. **清空旧 pending：**
   ```python
   working.pending_plate = ""
   working.pending_commands = []
   ```

2. **预执行新 action：**
   ```python
   new_edit_result = apply_plate_edit_commands(working.car_plate, response.commands)
   ```

3. **格式校验：** `is_valid_plate_number(new_edit_result.car_plate)`

---

##### B1-分支3-1：新修改格式不合规

**状态变更：** 无（pending 已清空，原车牌不变）

**回复话术：**
```python
assistant_reply = EDIT_INVALID_REPLY.format(plate=working.car_plate)
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "assistant_reply": "按这次修改后车牌格式不符合规则..."
}
```

**LLM 调用次数：** 1 次  
**下一轮：** pending_plate 为空 → 回到【分支 A】

---

##### B1-分支3-2：新修改格式合规，保存为新 pending

**状态变更：**
```python
working.pending_plate = new_edit_result.car_plate
working.pending_commands = [cmd.to_dict() for cmd in response.commands]
# working.car_plate 仍保持原车牌
```

**回复话术：**
```python
new_action_desc = describe_edit_commands(response.commands)
assistant_reply = PENDING_ACTION_CONFIRM_TEMPLATE.format(
    old_plate=working.car_plate,
    action_desc=new_action_desc,
    new_plate=working.pending_plate
)
```

**输出JSON：**
```json
{
  "task_status": "need_confirmation",
  "car_plate": "京AB123",
  "pending_plate": "京CB123",
  "assistant_reply": "当前暂存车牌是京AB123，识别到您要将第2位替换为C，修改后车牌京CB123，是否执行？"
}
```

**LLM 调用次数：** 1 次  
**下一轮：** pending_plate 不为空 → 再次进入【分支 B】

---

## LLM 调用次数汇总

| 场景 | LLM 调用 | 说明 |
|---|---|---|
| 分支 A：确认整车牌 | **1** | detect_confirmation |
| 分支 A：给修改意见 - 听不清 | **2** | detect_confirmation + update_car_plate |
| 分支 A：给修改意见 - 格式不合规 | **2** | detect_confirmation + update_car_plate |
| 分支 A：给修改意见 - 成功保存 pending | **2** | detect_confirmation + update_car_plate |
| 分支 B：确认执行 | **1** | classify_pending_response |
| 分支 B：拒绝执行无新意见 | **1** | classify_pending_response |
| 分支 B：拒绝执行+新意见（不合规） | **1** | classify_pending_response |
| 分支 B：拒绝执行+新意见（新 pending） | **1** | classify_pending_response |

**最多调用：2 次（分支 A 的修改轮）**  
**平均调用：约 1.5 次**

---

## 关键改动点

1. **移除：** `refresh_confusions_after_audio()` 在多轮路径中的所有调用
2. **新增：** `PlateAgentState.pending_plate` / `pending_commands` 字段
3. **新增：** `classify_pending_response()` 方法（分支 B 的 LLM 调用）
4. **新增：** `describe_edit_commands()` 函数（命令转中文描述）
5. **新增：** 3 条话术模板（PENDING_ACTION_*）
6. **修改：** `plate_agent.py` 多轮分支逻辑（增加 pending_plate 判断入口）

---

## 实现步骤

1. **步骤 1：** 更新 `PlateAgentState` 数据结构（plate_agent_types.py）
2. **步骤 2：** 实现命令描述函数（plate_agent_messages.py）
3. **步骤 3：** 新增话术模板（plate_agent_messages.py）
4. **步骤 4：** 实现 `classify_pending_response()` 方法（plate_agent_nodes.py）
5. **步骤 5：** 重写 `plate_agent.py` 多轮分支逻辑
6. **步骤 6：** 更新状态序列化/反序列化逻辑（plate_agent_state.py）
7. **步骤 7：** 测试验证各分支流程
