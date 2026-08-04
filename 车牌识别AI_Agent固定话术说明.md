# 语音车牌识别 AI Agent 固定话术文档

> 文档版本：2026-07-31 | 维护文件：`realtime_audio_demo/services/plate_agent_messages.py`
>
> **说明：** 所有话术集中维护于 `plate_agent_messages.py`，修改话术**只需修改该文件中等号右侧的字符串**，不需要修改任何流程逻辑代码。花括号 `{plate}` `{char}` `{pending}` 是后端占位符，**必须保留**。

---

## 一、话术总览索引

| 场景 | 变量名 | 触发时机 |
|------|--------|---------|
| 开场白 | `SESSION_OPENING_TEXT` | 会话刚开始时 |
| ACK 衔接语 | `ACK_MESSAGES_BY_SCENE` | 大模型处理期间（0/1/3/5秒） |
| 未识别到车牌 | `NO_PLATE_REPLY` | 首轮无车牌内容 |
| 车牌格式无效 | `INVALID_PLATE_REPLY` | 识别出来但不合法 |
| 编辑指令不清晰 | `EDIT_UNCLEAR_REPLY` | 无法解析修改意图 |
| 编辑后格式非法 | `EDIT_INVALID_REPLY` | 修改后车牌不合规 |
| 多步修改部分完成 | `EDIT_MULTI_STEP_PARTIAL_REPLY` | 多步修改未全部完成 |
| 保留当前车牌 | `EDIT_KEEP_CURRENT_PLATE_TEMPLATE` | 用户未实际修改 |
| 字符不存在 | `EDIT_CHAR_NOT_FOUND_TEMPLATE` | 要改的字符不在车牌里 |
| 字符重复歧义 | `EDIT_DUPLICATE_CHAR_TEMPLATE` | 同字符出现多次不确定改哪个 |
| 首轮识别成功 | `INITIAL_SUCCESS_TEMPLATE` | 首次识别到有效车牌 |
| 更新成功 | `UPDATE_SUCCESS_TEMPLATE` | 纠错修改成功 |
| 部分确认 | `PARTIAL_CONFIRMATION_TEMPLATE` | 确认了部分待确认字符 |
| 整车确认 | `CONFIRMED_REPLY_TEMPLATE` | 用户确认整车牌 |
| 待确认追加 | `PENDING_CONFIRMATION_TEMPLATE` | 追加到基础话术后面 |
| 无待确认追加 | `NO_PENDING_CONFIRMATION_TEMPLATE` | 无易混淆字符时追加 |
| 新能源追加 | `NEW_ENERGY_CONFIRMATION_TEXT` | 识别为8位新能源时追加 |

---

## 二、话术详细说明

### 2.1 开场白

```
SESSION_OPENING_TEXT = "您好，请告诉我您的车牌号。"
```

**触发：** `/api/chatbox/audio/session/start` 接口被调用时  
**用户体验：** 页面加载时 AI 主动提示用户说车牌  
**修改提示：** 可以自由修改文案，无占位符

---

### 2.2 ACK 衔接语（按时间点）

大模型处理期间，按 0秒→1秒→3秒→5秒 发出，结果就绪后停止。

#### 首轮（`initial`）场景 — 用户首次说车牌

| 时间点 | 当前话术 | 作用 |
|--------|---------|------|
| 0秒 | `"语音已收到，正在判断是否包含车牌信息。"` | 告知音频已到达 |
| 1秒 | `"正在识别车牌号码内容。"` | 识别进行中 |
| 3秒 | `"还在结合车牌规则和发音做确认。"` | 规则复核中 |
| 5秒 | `"识别还在处理，请稍等。"` | 长耗时兜底 |

#### 多轮（`update`）场景 — 已有暂存车牌，用户在确认/纠错

| 时间点 | 当前话术 | 作用 |
|--------|---------|------|
| 0秒 | `"语音已收到，正在判断您是在确认还是修改。"` | 告知已收到 |
| 1秒 | `"正在结合当前车牌处理您的这次回复。"` | 处理中 |
| 3秒 | `"还在复核修改结果和需要确认的位置。"` | 复核中 |
| 5秒 | `"处理还在继续，请稍等。"` | 长耗时兜底 |

**修改提示：** 只修改每条字符串，4条顺序不能调换，对应4个时间点。

---

### 2.3 错误与兜底话术

#### 未识别到车牌
```
NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"
```
**触发：** 首轮音频中模型未能提取任何车牌信息  
**典型场景：** 用户说的不是车牌、音频为空、噪声过大

---

#### 车牌格式无效
```
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"
```
**触发：** 模型提取了内容，但 `is_valid_plate_number()` 校验失败  
**典型场景：** 长度不是7位或8位、首位不是省份简称、字符包含非法字符  
**修改提示：** 无占位符，可自由修改

---

#### 编辑指令不清晰
```
EDIT_UNCLEAR_REPLY = "我没有听清您要修改车牌的哪一处，当前仍保留原来的车牌。请您说明要替换、插入或删除哪一位。"
```
**触发：** 多轮纠错时，模型无法形成有效编辑动作  
**典型场景：** 用户只说"不对"，没说改哪一位；语音模糊；操作类型不明确  
**修改提示：** 无占位符，可自由修改

---

#### 编辑后格式非法
```
EDIT_INVALID_REPLY = "按这次修改后车牌格式不符合规则，当前仍保留车牌{plate}。请您重新说明要改哪一处。"
```
**触发：** 编辑命令执行后，新车牌 `is_valid_plate_number()` = False  
**典型场景：** 删除/插入导致位数不合法  
**占位符：** `{plate}` = 执行失败后保留的旧车牌  
**修改提示：** `{plate}` 必须保留

---

#### 多步修改部分完成
```
EDIT_MULTI_STEP_PARTIAL_REPLY = "这次修改包含多步内容，我先处理到目前能确定的位置，请您继续确认或说明剩余要改的部分。"
```
**触发：** 用户一句话含多步修改，Agent 只成功执行了部分步骤  
**修改提示：** 无占位符，可自由修改

---

#### 保留当前车牌（无实际修改）
```
EDIT_KEEP_CURRENT_PLATE_TEMPLATE = "当前仍保留原来的车牌{plate}，请您确认是否正确。"
```
**触发：** 模型 action=none，或用户表达确认意图但整车尚未确认完成  
**占位符：** `{plate}` = 当前暂存车牌  
**生成函数：** `build_keep_current_plate_reply(plate)`

---

#### 字符不在车牌中
```
EDIT_CHAR_NOT_FOUND_TEMPLATE = "当前车牌里没有{char}，所以没有修改。当前仍保留原来的车牌。"
```
**触发：** 用户说"把 R 改成 2"，但当前车牌里没有 R  
**占位符：** `{char}` = 用户指定的字符口语描述（如"字母 R"）  
**生成函数：** `build_char_not_found_reply(value)` → `describe_plate_char(value)` 转为口语

---

#### 字符重复歧义
```
EDIT_DUPLICATE_CHAR_TEMPLATE = "当前车牌里有多个{char}，请您说明要改前面的还是后面的。"
```
**触发：** 用户说"把1改成E"，但车牌里有两个1，系统不知道改哪个  
**占位符：** `{char}` = 字符口语描述  
**生成函数：** `build_duplicate_char_reply(value)`  
**解决方式：** 用户说"前面的1"或"第3位"

---

### 2.4 成功话术（基础模板）

#### 首轮识别成功
```
INITIAL_SUCCESS_TEMPLATE = "我识别到的车牌号是{plate}。"
```
**后续拼接：** 自动追加 `PENDING_CONFIRMATION_TEMPLATE` 或 `NO_PENDING_CONFIRMATION_TEMPLATE`  
**占位符：** `{plate}` = 首轮识别出的完整车牌  
**生成函数：** `build_initial_success_reply(state)`

完整示例输出：
> "我识别到的车牌号是津A12345。需要您确认：第1位是天津的津，第3位是数字1。"

---

#### 纠错更新成功
```
UPDATE_SUCCESS_TEMPLATE = "已按您的说明更新为{plate}。"
```
**后续拼接：** 自动追加确认内容  
**占位符：** `{plate}` = 修改后的完整车牌  
**生成函数：** `build_update_success_reply(state)`

完整示例输出：
> "已按您的说明更新为津AE2345。需要您确认：第1位是天津的津。"

---

#### 部分字符确认
```
PARTIAL_CONFIRMATION_TEMPLATE = "好的，已记录您对车牌{plate}的确认。"
```
**后续拼接：** 追加剩余待确认内容；若无剩余则追加整车确认提示  
**占位符：** `{plate}` = 当前暂存车牌  
**生成函数：** `build_partial_confirmation_reply(state)`

完整示例输出：
> "好的，已记录您对车牌津A12345的确认。需要您确认：第1位是天津的津。"

---

#### 整车确认完成
```
CONFIRMED_REPLY_TEMPLATE = "好的，已确认您的车牌号是{plate}。"
```
**触发：** 用户明确确认整车牌（"对/正确/没问题"）且本轮未执行编辑  
**占位符：** `{plate}` = 最终车牌号  
**生成函数：** `build_confirmed_reply(plate)`  
**注意：** 这是任务结束话术，后续无追加内容

---

### 2.5 确认内容追加模板

#### 有待确认字符时（追加到基础话术后）
```
PENDING_CONFIRMATION_TEMPLATE = "需要您确认：{pending}。"
```
**占位符：** `{pending}` = 待确认字符描述列表，格式为"第N位是XXX，第M位是YYY"  
**生成函数：** `pending_confirmation_text(state)` → `pending_confirmation_descriptions(state)`

描述生成规则：`describe_plate_char(value)` 转换 + 位置标注
> 示例："{pending}" = "第1位是天津的津，第4位是数字2"

---

#### 无待确认字符时（追加到基础话术后）
```
NO_PENDING_CONFIRMATION_TEMPLATE = "请您确认{plate}是否正确。"
```
**占位符：** `{plate}` = 当前暂存车牌  
**触发：** `need_confirm_chars` 为空时，但整车还未最终确认

---

#### 新能源车牌追加
```
NEW_ENERGY_CONFIRMATION_TEXT = "另外这是新能源号牌吗？"
```
**触发：** `state.vehicle_type == "new_energy"`（识别为8位车牌）  
**追加位置：** 在所有确认内容之后  
**修改提示：** 无占位符，可自由修改

---

## 三、话术拼接规则

话术拼接由 `with_pending_confirmation()` 函数统一完成：

```
基础话术 → ensure_sentence()（确保以句号结尾）
         → 若有 need_confirm_chars：追加 PENDING_CONFIRMATION_TEMPLATE
         → 若无 need_confirm_chars：追加 NO_PENDING_CONFIRMATION_TEMPLATE
         → 若 vehicle_type == "new_energy"：追加 NEW_ENERGY_CONFIRMATION_TEXT
```

**拼接示例（首轮识别，有1位易混淆）：**
```
"我识别到的车牌号是津A12345。" + "需要您确认：第1位是天津的津。"
→ "我识别到的车牌号是津A12345。需要您确认：第1位是天津的津。"
```

**拼接示例（新能源，无待确认）：**
```
"我识别到的车牌号是沪A1234E。" + "请您确认沪A1234E是否正确。" + "另外这是新能源号牌吗？"
→ "我识别到的车牌号是沪A1234E。请您确认沪A1234E是否正确。另外这是新能源号牌吗？"
```

---

## 四、话术触发时机完整映射

```
用户发起会话
  └─► SESSION_OPENING_TEXT（"您好，请告诉我您的车牌号。"）

大模型处理中（0/1/3/5秒）
  ├─ 首轮 └─► ACK_MESSAGES_BY_SCENE["initial"][0~3]
  └─ 多轮 └─► ACK_MESSAGES_BY_SCENE["update"][0~3]

首轮结果返回
  ├─ 无车牌内容     └─► NO_PLATE_REPLY
  ├─ 车牌格式无效   └─► INVALID_PLATE_REPLY
  └─ 识别成功       └─► INITIAL_SUCCESS_TEMPLATE + 确认追加

多轮结果返回
  ├─ 编辑成功       └─► UPDATE_SUCCESS_TEMPLATE + 确认追加
  ├─ 部分字符确认   └─► PARTIAL_CONFIRMATION_TEMPLATE + 确认追加
  ├─ 整车确认       └─► CONFIRMED_REPLY_TEMPLATE（任务结束）
  ├─ 编辑不清晰     └─► EDIT_UNCLEAR_REPLY + 确认追加
  ├─ 编辑后非法     └─► EDIT_INVALID_REPLY + 确认追加
  ├─ 保留原车牌     └─► EDIT_KEEP_CURRENT_PLATE_TEMPLATE
  ├─ 字符找不到     └─► EDIT_CHAR_NOT_FOUND_TEMPLATE
  └─ 字符重复歧义   └─► EDIT_DUPLICATE_CHAR_TEMPLATE
```

---

## 五、快速修改指引

**修改文件：** `realtime_audio_demo/services/plate_agent_messages.py`

1. **只改文案内容**：直接修改等号右侧字符串
2. **不改变量名**：`SESSION_OPENING_TEXT`、`ACK_MESSAGES_BY_SCENE` 等名称不能修改
3. **保留占位符**：`{plate}`、`{char}`、`{pending}` 必须原样保留
4. **ACK 话术**：每个场景（initial/update）固定4条，顺序对应0/1/3/5秒，不能增删

**测试建议：** 修改后，重点验证以下场景：
- 首轮识别到易混淆车牌（如津开头、含数字1/2和字母E/R）
- 用户说"第3位改成E"（`replace_position`路径）
- 用户说"对的"（整车确认路径）
- 8位新能源车牌（新能源追加话术）
