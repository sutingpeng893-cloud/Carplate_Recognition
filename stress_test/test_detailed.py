#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车牌识别系统详细延迟压力测试脚本
基于 test.py，增加完整的延迟指标输出，并对每个指标添加清晰说明。

【延迟指标含义说明】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ack_time_ms（过渡话术到达延迟）
  含义：从客户端发出请求 → 收到第一个 SSE "ack" 事件的客户端侧时间。
  "ack" 是服务器在固定时间点主动推送的过渡话术（如"稍等一下"），不包含任何
  大模型调用或业务处理。当前服务器配置：请求到达约 1 秒后发出。
  组成：网络往返延迟（RTT）+ 服务器 ack 定时器（~1000ms）
  ★ 该指标反映用户听到"稍等"过渡话术需要等多久，不代表识别结果速度。

result_time_ms（识别结果到达延迟）
  含义：从客户端发出请求 → 收到 SSE "result" 事件的客户端侧时间。
  "result" 事件包含本轮完整 Agent 输出：识别/纠错后的候选车牌 + 回复给
  用户的完整话术（如"京A·12B45，第3位B是否正确？"，约15~25字固定模板）。
  组成：网络往返延迟 + 全部大模型调用耗时 + 业务处理 + 回复生成
  ★ 这是用户前端真正看到第一段文字的时刻，是最重要的用户体验指标。

ack_to_result_gap_ms（大模型推理耗时估算）
  含义：ack 事件送达用户 → result 事件送达用户的间隔时间。
  计算：result_time_ms - ack_time_ms
  组成：本轮所有大模型调用 + 工具执行 + 业务逻辑处理的实际总耗时（近似）。
  ★ 该值越小说明模型推理越快，是优化大模型调用效率的核心参考指标。
  首轮第一个 LLM 调用：detect_plate_presence（仅输出 true/false，max_tokens=8）
  多轮第一个 LLM 调用：detect_confirmation（仅输出 yes/no，max_tokens=8）

total_time_ms（请求完整生命周期）
  含义：从客户端发出请求 → SSE 流完全关闭（[DONE]）的总时间。
  包含：所有 ack / result 事件 + TTS 生成 + 后处理 + 连接关闭。
  ★ 反映服务端完成本次请求全部工作所需时间。

latency_ms（与服务端日志一致）
  含义：等同于服务端 JSON 日志中的 latency_ms，即服务端自报的本轮处理总耗时。
  来源：result 事件携带的 latency_ms 字段，由服务端测量，不含网络 RTT。
  ★ 与日志中 ai_inference_metadata.latency_ms 含义相同，可直接对比。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


METRIC_LEGEND = {
    "ack_time_ms": "固定过渡话术到达延迟 | 请求发出→收到'稍等'固定话术 | ≈网络RTT+ack定时器(~1000ms) | 与大模型无关，是硬编码字符串",
    "first_text_time_ms": "LLM首字到达延迟(TTFT) | 第一个含实际文字的非ack事件到达时间 | 当前架构下=result_time_ms(回复一次性到达)；流式改造后将变为真正TTFT",
    "result_time_ms": "完整回复到达延迟 | 请求发出→收到含车牌识别结果的完整话术 | 含全部LLM调用+业务处理 | 用户看到文字的真实时刻",
    "ack_to_result_gap_ms": "大模型推理耗时估算 | result到达时间-ack到达时间 | 近似本轮所有LLM调用串行耗时；越小说明推理越快",
    "total_time_ms": "请求完整生命周期 | 请求发出→SSE流关闭 | 含TTS+后处理+连接关闭",
    "latency_ms": "服务端自报耗时 | 与服务端日志 ai_inference_metadata.latency_ms 含义一致 | 不含网络RTT | 等同于 server_latency_ms",
    "server_latency_ms": "服务端自报延迟 | 服务器在result事件中上报的处理耗时 | 纯服务端视角，不含网络RTT；result_time_ms - server_latency_ms ≈ 客户端到服务端的网络往返时延(RTT)",
}


def encode_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_first_turn_filename(audio_dir: Path) -> str:
    """Auto-detect the first turn filename from the audio directory."""
    for candidate in ("turn_000_real.wav", "turn_000.wav"):
        if any((d / "audio" / candidate).exists() for d in audio_dir.iterdir() if d.is_dir()):
            return candidate
    return "turn_000.wav"


def get_turn_files(session_dir: Path) -> list:
    audio_dir = session_dir / "audio"
    if not audio_dir.exists():
        return []
    return sorted([f for f in audio_dir.iterdir() if f.name.startswith("turn_") and f.suffix == ".wav"])


def test_session_turns_stream(
    session_id: str,
    session_dir: Path,
    base_url: str,
    model: str,
    timeout: int = 120,
    first_turn_file: str = "turn_000.wav",
) -> list:
    results = []
    turn_files = get_turn_files(session_dir)
    if not turn_files:
        return results

    for turn_file in turn_files:
        turn_name = turn_file.name
        is_first_turn = turn_name == first_turn_file

        try:
            audio_base64 = encode_audio(str(turn_file))
            payload = {
                "session_id": session_id,
                "model": model,
                "audio_base64": audio_base64,
                "outputAudio": False,
            }

            total_start = time.perf_counter()
            events_with_timing = []

            with httpx.stream("POST", f"{base_url}/api/chatbox/audio/stream", json=payload, timeout=timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            event_time_ms = int((time.perf_counter() - total_start) * 1000)
                            events_with_timing.append({"time_ms": event_time_ms, "event": data})
                        except json.JSONDecodeError:
                            pass

            total_time_ms = int((time.perf_counter() - total_start) * 1000)

            ack_time_ms = None
            result_time_ms = None
            result_data = {}
            # first_text_time_ms: 第一个携带实际文字内容（非 ack 固定话术）的事件到达时间。
            # 当前架构下所有 LLM 输出一次性打包在 result 事件里，所以等于 result_time_ms。
            # 若将来改为流式吐字，此指标将自动变为真正的 TTFT（Time To First Token）。
            first_text_time_ms = None
            first_text_chars = ""

            for item in events_with_timing:
                stage = item["event"].get("stage")
                text = (item["event"].get("speech_text") or item["event"].get("text") or "").strip()
                if stage == "ack" and ack_time_ms is None:
                    ack_time_ms = item["time_ms"]
                elif stage == "result":
                    result_time_ms = item["time_ms"]
                    result_data = item["event"]
                # 第一个非 ack 且含文字的事件 → LLM 输出首字到达时刻
                if text and stage != "ack" and first_text_time_ms is None:
                    first_text_time_ms = item["time_ms"]
                    first_text_chars = text[:5]

            server_latency = result_data.get("latency_ms", 0)
            agent_state = result_data.get("agent_state", {})
            # llm_calls: 服务端在 result 事件里返回的每次 audio_call 明细
            # 格式：[{node, duration_ms, max_tokens, output_chars}, ...]
            llm_calls_raw = result_data.get("llm_calls") or []
            latency_ms = server_latency  # 与服务端日志 latency_ms 含义一致
            ack_to_result_gap_ms = (result_time_ms - ack_time_ms) if ack_time_ms and result_time_ms else None

            # 按 node 名汇总各 LLM 函数耗时（一轮内可能多次调用同一 node）
            llm_node_times: dict[str, list[int]] = {}
            for call in llm_calls_raw:
                n = call.get("node") or "unknown"
                d = call.get("duration_ms")
                if isinstance(d, (int, float)):
                    llm_node_times.setdefault(n, []).append(int(d))

            results.append({
                "session_id": session_id,
                "turn": turn_name,
                "turn_type": "first_turn" if is_first_turn else "multi_turn",
                "status": "success",
                "ack_time_ms": ack_time_ms,
                "result_time_ms": result_time_ms,
                "first_text_time_ms": first_text_time_ms,
                "first_text_chars": first_text_chars,
                "ack_to_result_gap_ms": ack_to_result_gap_ms,
                "total_time_ms": total_time_ms,
                "latency_ms": latency_ms,
                "server_latency_ms": server_latency,
                "has_ack": ack_time_ms is not None,
                "car_plate": agent_state.get("car_plate", ""),
                "final_car_plate": agent_state.get("final_car_plate", ""),
                "assistant_reply": result_data.get("speech_text", "")[:120],
                "events_count": len(events_with_timing),
                "llm_calls": llm_calls_raw,
                "llm_node_times": llm_node_times,
            })

        except Exception as e:  # noqa: BLE001
            results.append({
                "session_id": session_id,
                "turn": turn_name,
                "turn_type": "first_turn" if is_first_turn else "multi_turn",
                "status": "error",
                "error": str(e)[:200],
            })

    return results


def calc_percentile(data, percentile):
    if not data:
        return 0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = (n - 1) * percentile / 100
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return round(sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight, 2)


def calc_stats(data):
    if not data:
        return {"avg": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "count": 0}
    return {
        "avg": round(sum(data) / len(data), 2),
        "p25": calc_percentile(data, 25),
        "p50": calc_percentile(data, 50),
        "p75": calc_percentile(data, 75),
        "p95": calc_percentile(data, 95),
        "count": len(data),
    }


def run_single_round(
    sessions: list,
    base_url: str,
    model: str,
    concurrency_level: int,
    round_num: int,
    timeout: int = 120,
    launch_window: float = 1.0,
    first_turn_file: str = "turn_000.wav",
) -> tuple[list, dict]:
    results = []
    launch_interval = launch_window / len(sessions) if launch_window > 0 and sessions else 0

    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = {}
        round_launch_start = time.perf_counter()
        for idx, session_dir in enumerate(sessions):
            if launch_interval > 0:
                scheduled_at = round_launch_start + idx * launch_interval
                wait_seconds = scheduled_at - time.perf_counter()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            session_id_base = session_dir.name
            session_id = f"{session_id_base}_{concurrency_level}_{round_num}_{int(time.time() * 1000)}_{idx}"
            future = executor.submit(test_session_turns_stream, session_id, session_dir, base_url, model, timeout, first_turn_file)
            futures[future] = session_id

        for future in as_completed(futures):
            results.extend(future.result())

    first_turn_results = [r for r in results if r.get("turn_type") == "first_turn" and r.get("status") == "success"]
    multi_turn_results = [r for r in results if r.get("turn_type") == "multi_turn" and r.get("status") == "success"]

    def extract(rows, key):
        return [r[key] for r in rows if r.get(key) is not None]

    def extract_node_times(rows: list, node: str) -> list[int]:
        """从 llm_node_times 字典中提取特定 node 的所有耗时样本。"""
        out = []
        for r in rows:
            for ms in (r.get("llm_node_times") or {}).get(node, []):
                out.append(ms)
        return out

    # 首轮/多轮各 LLM node 的耗时统计
    # node 名称来自 plate_agent_nodes.py 中 audio_call(..., node=xxx) 的传参
    first_turn_llm_nodes = [
        "detect_plate_presence",
        "extract_car_plate",
        "normalize_plate_result",
        "detect_confirmation_state_actions",
    ]
    multi_turn_llm_nodes = [
        "detect_confirmation",
        "update_car_plate.edit_command",
        "review_plate_update",
        "classify_intent",
        "classify_pending_commands",
        "detect_confirmation_state_actions",
    ]

    stats = {
        "concurrency": concurrency_level,
        "round": round_num,
        "launch_window_seconds": launch_window,
        "launch_interval_seconds": round(launch_interval, 4),
        "total_sessions": len(sessions),
        "total_turns": len(results),
        "success_count": sum(1 for r in results if r.get("status") == "success"),
        "fail_count": sum(1 for r in results if r.get("status") in ["fail", "error", "timeout"]),
        "first_turn_count": len([r for r in results if r.get("turn_type") == "first_turn"]),
        "first_turn_success": len(first_turn_results),
        "multi_turn_count": len([r for r in results if r.get("turn_type") == "multi_turn"]),
        "multi_turn_success": len(multi_turn_results),
        # 首轮端到端延迟
        "first_turn_ack_time":        calc_stats(extract(first_turn_results, "ack_time_ms")),
        "first_turn_result_time":     calc_stats(extract(first_turn_results, "result_time_ms")),
        "first_turn_first_text_time": calc_stats(extract(first_turn_results, "first_text_time_ms")),
        "first_turn_ack_gap":         calc_stats(extract(first_turn_results, "ack_to_result_gap_ms")),
        "first_turn_total_time":      calc_stats(extract(first_turn_results, "total_time_ms")),
        "first_turn_latency":         calc_stats(extract(first_turn_results, "latency_ms")),
        "first_turn_server_latency":  calc_stats(extract(first_turn_results, "server_latency_ms")),
        # 首轮各 LLM node 耗时（来自服务端 llm_calls 字段）
        **{f"first_turn_llm_{n.replace('.', '_')}": calc_stats(extract_node_times(first_turn_results, n))
           for n in first_turn_llm_nodes},
        # 多轮端到端延迟
        "multi_turn_ack_time":        calc_stats(extract(multi_turn_results, "ack_time_ms")),
        "multi_turn_result_time":     calc_stats(extract(multi_turn_results, "result_time_ms")),
        "multi_turn_first_text_time": calc_stats(extract(multi_turn_results, "first_text_time_ms")),
        "multi_turn_ack_gap":         calc_stats(extract(multi_turn_results, "ack_to_result_gap_ms")),
        "multi_turn_total_time":      calc_stats(extract(multi_turn_results, "total_time_ms")),
        "multi_turn_latency":         calc_stats(extract(multi_turn_results, "latency_ms")),
        "multi_turn_server_latency":  calc_stats(extract(multi_turn_results, "server_latency_ms")),
        # 多轮各 LLM node 耗时
        **{f"multi_turn_llm_{n.replace('.', '_')}": calc_stats(extract_node_times(multi_turn_results, n))
           for n in multi_turn_llm_nodes},
    }
    return results, stats


def run_concurrency_with_multiple_rounds(
    sessions: list,
    base_url: str,
    model: str,
    concurrency_level: int,
    num_rounds: int,
    timeout: int = 120,
    launch_window: float = 1.0,
    first_turn_file: str = "turn_000.wav",
) -> tuple[list, list]:
    all_results = []
    all_stats = []
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}/{num_rounds}...", end=" ", flush=True)
        start = time.time()
        start_idx = (round_num - 1) * concurrency_level
        round_sessions = sessions[start_idx: start_idx + concurrency_level]
        results, stats = run_single_round(
            round_sessions, base_url, model, concurrency_level, round_num, timeout, launch_window, first_turn_file
        )
        duration = time.time() - start
        all_results.extend(results)
        all_stats.append(stats)
        print(f"OK ({stats['success_count']}/{stats['total_turns']}) - {duration:.1f}s")
    return all_results, all_stats


def merge_stats_across_rounds(stats_list: list) -> dict:
    """Merge per-round stats into a single aggregated stat dict."""
    if not stats_list:
        return {}

    def merge_key(key):
        all_vals = []
        for s in stats_list:
            sub = s.get(key, {})
            count = sub.get("count", 0)
            avg = sub.get("avg", 0)
            if count > 0 and avg > 0:
                all_vals.extend([avg] * count)
        return calc_stats(all_vals) if all_vals else {"avg": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "count": 0}

    keys = [
        "first_turn_ack_time", "first_turn_result_time", "first_turn_first_text_time", "first_turn_ack_gap",
        "first_turn_total_time", "first_turn_latency", "first_turn_server_latency",
        "first_turn_llm_detect_plate_presence",
        "first_turn_llm_extract_car_plate",
        "first_turn_llm_normalize_plate_result",
        "first_turn_llm_detect_confirmation_state_actions",
        "multi_turn_ack_time", "multi_turn_result_time", "multi_turn_first_text_time", "multi_turn_ack_gap",
        "multi_turn_total_time", "multi_turn_latency", "multi_turn_server_latency",
        "multi_turn_llm_detect_confirmation",
        "multi_turn_llm_update_car_plate_edit_command",
        "multi_turn_llm_review_plate_update",
        "multi_turn_llm_classify_intent",
        "multi_turn_llm_classify_pending_commands",
        "multi_turn_llm_detect_confirmation_state_actions",
    ]
    merged = {}
    for k in keys:
        merged[k] = merge_key(k)
    merged["concurrency"] = stats_list[0]["concurrency"]
    merged["total_sessions"] = sum(s["total_sessions"] for s in stats_list)
    merged["total_turns"] = sum(s["total_turns"] for s in stats_list)
    merged["success_count"] = sum(s["success_count"] for s in stats_list)
    merged["fail_count"] = sum(s["fail_count"] for s in stats_list)
    return merged


def fmt(v) -> str:
    """Format a float/int value for table display."""
    if v is None or v == 0:
        return "  -   "
    return f"{v:7.1f}"


def print_concurrency_table(merged: dict) -> None:
    """Print a single-concurrency detailed table to stdout."""
    c = merged["concurrency"]
    ok = merged["success_count"]
    fail = merged["fail_count"]
    total = merged["total_turns"]

    print(f"\n  并发={c}  成功={ok}/{total}  失败={fail}")
    print(f"  {'指标':<36}  {'avg':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p95':>7}")
    print(f"  {'-'*36}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    rows = [
        ("首轮 ack_time_ms      [固定过渡话术到达]",   "first_turn_ack_time"),
        ("首轮 first_text_ms    [LLM首字到达/TTFT]",  "first_turn_first_text_time"),
        ("首轮 result_time_ms   [完整回复到达]",       "first_turn_result_time"),
        ("首轮 ack_to_result    [LLM推理耗时估算]",    "first_turn_ack_gap"),
        ("首轮 total_time_ms    [请求完整生命周期]",   "first_turn_total_time"),
        ("首轮 server_latency   [服务端自报耗时]",     "first_turn_server_latency"),
        ("",                                           None),
        ("首轮 ▸detect_plate_presence [有无车牌判断]", "first_turn_llm_detect_plate_presence"),
        ("首轮 ▸extract_car_plate     [车牌提取]",     "first_turn_llm_extract_car_plate"),
        ("首轮 ▸normalize_plate_result[省份归一化]",   "first_turn_llm_normalize_plate_result"),
        ("首轮 ▸confirm_state_actions [确认状态更新]", "first_turn_llm_detect_confirmation_state_actions"),
        ("",                                           None),
        ("多轮 ack_time_ms      [固定过渡话术到达]",   "multi_turn_ack_time"),
        ("多轮 first_text_ms    [LLM首字到达/TTFT]",  "multi_turn_first_text_time"),
        ("多轮 result_time_ms   [完整回复到达]",       "multi_turn_result_time"),
        ("多轮 ack_to_result    [LLM推理耗时估算]",    "multi_turn_ack_gap"),
        ("多轮 total_time_ms    [请求完整生命周期]",   "multi_turn_total_time"),
        ("多轮 server_latency   [服务端自报耗时]",     "multi_turn_server_latency"),
        ("",                                           None),
        ("多轮 ▸detect_confirmation   [是否整体确认]", "multi_turn_llm_detect_confirmation"),
        ("多轮 ▸update_car_plate.edit [提取纠错命令]", "multi_turn_llm_update_car_plate_edit_command"),
        ("多轮 ▸review_plate_update   [编辑复核]",     "multi_turn_llm_review_plate_update"),
        ("多轮 ▸classify_intent       [待确认意图分类]","multi_turn_llm_classify_intent"),
        ("多轮 ▸classify_pending_cmds [提取编辑命令]", "multi_turn_llm_classify_pending_commands"),
        ("多轮 ▸confirm_state_actions [确认状态更新]", "multi_turn_llm_detect_confirmation_state_actions"),
    ]

    for label, key in rows:
        if key is None:
            print()
            continue
        s = merged.get(key, {})
        print(
            f"  {label:<36}  "
            f"{fmt(s.get('avg')):>7}  "
            f"{fmt(s.get('p25')):>7}  "
            f"{fmt(s.get('p50')):>7}  "
            f"{fmt(s.get('p75')):>7}  "
            f"{fmt(s.get('p95')):>7}"
        )


def _fmt_stat(stat: dict, key: str) -> str:
    val = stat.get(key, 0)
    return f"{val:.1f}" if val else "-"


def build_markdown_report(all_merged: list[dict], base_url: str, model: str, timestamp: str) -> str:
    lines = []
    lines.append("# 车牌识别压力测试延迟详细报告")
    lines.append(f"\n测试时间：{timestamp}  ")
    lines.append(f"服务地址：{base_url}  ")
    lines.append(f"模型：{model}  ")

    lines.append("\n---\n")
    lines.append("## 延迟指标说明\n")
    lines.append("| 指标 | 含义 | 组成 |")
    lines.append("|---|---|---|")
    lines.append('| **ack_time_ms** | 请求发出→收到过渡话术（如"稍等"）的时间 | 网络RTT + ack定时器（当前约1000ms），**不含大模型**，是固定硬编码话术 |')
    lines.append("| **first_text_time_ms** | 请求发出→第一个含实际文字的非ack SSE事件到达时间 | 当前架构下等于result_time_ms（回复一次性到达）；若改为流式吐字则自动变为真正TTFT |")
    lines.append("| **result_time_ms** | 请求发出→收到完整识别结果话术的时间 | 网络RTT + **全部LLM调用** + 业务处理 + 回复生成；用户看到文字的真实时刻 |")
    lines.append("| **ack_to_result_gap_ms** | result到达时间 − ack到达时间 | **近似本轮所有LLM调用+业务逻辑的实际总耗时**；优化推理效率的核心参考 |")
    lines.append("| **total_time_ms** | 请求发出→SSE流关闭 | 含TTS、后处理、连接关闭的完整生命周期 |")
    lines.append("| **server_latency_ms** | 服务器在result事件中自报的处理耗时 | 纯服务端视角，不含网络RTT；result_time_ms − server_latency_ms ≈ 网络往返时延(RTT) |")

    lines.append("\n---\n")
    lines.append("## 首轮延迟（First Turn）\n")
    lines.append("> 用户第一次说车牌，系统从无到有识别出候选车牌的过程。首轮LLM调用：detect_plate_presence(true/false) → extract_car_plate → normalize。\n")

    metrics = [
        ("ack_time_ms（固定过渡话术到达）",           "first_turn_ack_time"),
        ("first_text_time_ms（LLM首字到达/TTFT）",    "first_turn_first_text_time"),
        ("result_time_ms（完整回复到达）",             "first_turn_result_time"),
        ("ack_to_result_gap_ms（LLM推理耗时估算）",   "first_turn_ack_gap"),
        ("total_time_ms（请求完整生命周期）",          "first_turn_total_time"),
        ("server_latency_ms（服务端自报耗时）",        "first_turn_server_latency"),
    ]

    lines.append("| 并发数 | 指标 | avg (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for merged in all_merged:
        c = merged["concurrency"]
        for label, key in metrics:
            s = merged.get(key, {})
            lines.append(f"| {c} | {label} | {_fmt_stat(s,'avg')} | {_fmt_stat(s,'p25')} | {_fmt_stat(s,'p50')} | {_fmt_stat(s,'p75')} | {_fmt_stat(s,'p95')} |")

    lines.append("\n---\n")
    lines.append("## 多轮延迟（Multi-Turn）\n")
    lines.append("> 已有候选车牌后，用户确认或纠错的过程。多轮LLM调用：detect_confirmation(yes/no) → (若纠错) edit_command × N + review × N → refresh_confusions。\n")

    metrics_multi = [
        ("ack_time_ms（固定过渡话术到达）",           "multi_turn_ack_time"),
        ("first_text_time_ms（LLM首字到达/TTFT）",    "multi_turn_first_text_time"),
        ("result_time_ms（完整回复到达）",             "multi_turn_result_time"),
        ("ack_to_result_gap_ms（LLM推理耗时估算）",   "multi_turn_ack_gap"),
        ("total_time_ms（请求完整生命周期）",          "multi_turn_total_time"),
        ("server_latency_ms（服务端自报耗时）",        "multi_turn_server_latency"),
    ]

    lines.append("| 并发数 | 指标 | avg (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for merged in all_merged:
        c = merged["concurrency"]
        for label, key in metrics_multi:
            s = merged.get(key, {})
            lines.append(f"| {c} | {label} | {_fmt_stat(s,'avg')} | {_fmt_stat(s,'p25')} | {_fmt_stat(s,'p50')} | {_fmt_stat(s,'p75')} | {_fmt_stat(s,'p95')} |")

    lines.append("\n---\n")
    lines.append("## 首轮延迟（按指标排序）\n")
    lines.append("> 同一指标横向对比各并发级别的表现。\n")
    lines.append("| 指标 | 并发数 | avg (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, key in metrics:
        for merged in all_merged:
            c = merged["concurrency"]
            s = merged.get(key, {})
            lines.append(f"| {label} | {c} | {_fmt_stat(s,'avg')} | {_fmt_stat(s,'p25')} | {_fmt_stat(s,'p50')} | {_fmt_stat(s,'p75')} | {_fmt_stat(s,'p95')} |")

    lines.append("\n---\n")
    lines.append("## 多轮延迟（按指标排序）\n")
    lines.append("> 同一指标横向对比各并发级别的表现。\n")
    lines.append("| 指标 | 并发数 | avg (ms) | p25 (ms) | p50 (ms) | p75 (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, key in metrics_multi:
        for merged in all_merged:
            c = merged["concurrency"]
            s = merged.get(key, {})
            lines.append(f"| {label} | {c} | {_fmt_stat(s,'avg')} | {_fmt_stat(s,'p25')} | {_fmt_stat(s,'p50')} | {_fmt_stat(s,'p75')} | {_fmt_stat(s,'p95')} |")

    lines.append("\n---\n")
    lines.append("## 关键对比：ack_to_result_gap_ms（LLM推理耗时）\n")
    lines.append("> 该指标越小说明大模型推理越快，不受ACK定时器影响，是比较两个版本真实性能差异的正确指标。\n")
    lines.append("| 并发数 | 首轮 avg (ms) | 首轮 p50 (ms) | 首轮 p95 (ms) | 多轮 avg (ms) | 多轮 p50 (ms) | 多轮 p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for merged in all_merged:
        c = merged["concurrency"]
        fg = merged.get("first_turn_ack_gap", {})
        mg = merged.get("multi_turn_ack_gap", {})
        lines.append(
            f"| {c} | {_fmt_stat(fg,'avg')} | {_fmt_stat(fg,'p50')} | {_fmt_stat(fg,'p95')} "
            f"| {_fmt_stat(mg,'avg')} | {_fmt_stat(mg,'p50')} | {_fmt_stat(mg,'p95')} |"
        )

    return "\n".join(lines) + "\n"


def run_all_concurrency_tests(
    audio_dir: Path,
    base_url: str,
    model: str,
    concurrency_levels: list,
    num_rounds: int,
    timeout: int = 120,
    launch_window: float = 1.0,
    output_dir: str = "stress_test",
    first_turn_file: str = "",
) -> None:
    first_turn_name = first_turn_file or _detect_first_turn_filename(audio_dir)
    all_sessions = sorted(
        [d for d in audio_dir.iterdir() if d.is_dir() and (d / "audio" / first_turn_name).exists()],
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_path / f"detailed_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("车牌识别详细延迟压力测试")
    print(f"{'='*80}")
    print(f"服务地址：{base_url}")
    print(f"模型：{model}")
    print(f"并发级别：{concurrency_levels}")
    print(f"每并发轮数：{num_rounds}")
    print(f"超时：{timeout}s  启动窗口：{launch_window}s")
    print(f"首轮文件名：{first_turn_name}")
    print(f"可用会话数：{len(all_sessions)}")
    print(f"输出目录：{result_dir}")
    print("\n延迟指标说明：")
    print("  ack_time_ms       请求→过渡话术到达  ≈ 网络RTT + ack定时器(~1000ms)，不含大模型")
    print("  result_time_ms    请求→识别结果到达  含全部LLM调用，用户看到文字的真实时刻")
    print("  ack_to_result_gap result - ack 的差值，近似本轮LLM推理+业务处理总耗时")
    print("  total_time_ms     请求→SSE流关闭，含TTS+后处理")
    print(f"{'='*80}\n")

    all_merged = []

    for concurrency_level in concurrency_levels:
        total_needed = concurrency_level * num_rounds
        if total_needed > len(all_sessions):
            print(f"跳过并发={concurrency_level}：需要 {total_needed} 个会话，实际只有 {len(all_sessions)} 个")
            continue

        print(f"\n>>> 并发={concurrency_level}  ({concurrency_level} 会话/轮 × {num_rounds} 轮)")
        print(f"{'-'*80}")

        sessions = all_sessions[:total_needed]
        start_time = time.time()
        results, stats_list = run_concurrency_with_multiple_rounds(
            sessions, base_url, model, concurrency_level, num_rounds, timeout, launch_window, first_turn_name
        )
        duration = time.time() - start_time

        merged = merge_stats_across_rounds(stats_list)
        all_merged.append(merged)
        print_concurrency_table(merged)
        print(f"\n  总耗时：{duration:.1f}s")

        # 保存单并发结果
        concurrency_result = {
            "test_type": "detailed_latency_stress_test",
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "model": model,
            "concurrency_level": concurrency_level,
            "num_rounds": num_rounds,
            "metric_legend": METRIC_LEGEND,
            "merged_stats": merged,
            "statistics_by_round": stats_list,
            "all_results": results,
        }
        out_file = result_dir / f"concurrency_{concurrency_level}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(concurrency_result, f, ensure_ascii=False, indent=2)
        print(f"  已保存：{out_file}")

    # 汇总 JSON
    summary = {
        "test_type": "detailed_latency_stress_test_summary",
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url,
        "model": model,
        "concurrency_levels": concurrency_levels,
        "num_rounds": num_rounds,
        "metric_legend": METRIC_LEGEND,
        "summary_by_concurrency": all_merged,
    }
    summary_file = result_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    ts_label = datetime.now().isoformat(timespec="seconds")
    md_content = build_markdown_report(all_merged, base_url, model, ts_label)
    md_file = result_dir / "latency_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n{'='*80}")
    print("全部测试完成")
    print(f"汇总 JSON：{summary_file}")
    print(f"Markdown 报告：{md_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="车牌识别详细延迟压力测试（含 ack_to_result_gap_ms）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio-dir", "-d", default="stress_test/audio_data", help="音频数据目录")
    parser.add_argument("--base-url", "-u", default="http://127.0.0.1:55785", help="API 地址")
    parser.add_argument("--model", "-m", default="qwen3-omni", help="模型名称")
    parser.add_argument("--output-dir", "-o", default="stress_test/history", help="输出目录")
    parser.add_argument("--concurrency-levels", "-c", default="1,5,10,20,30,50", help="并发级别，逗号分隔")
    parser.add_argument("--num-rounds", "-r", type=int, default=20, help="每并发轮数")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="单请求超时秒数")
    parser.add_argument("--launch-window", type=float, default=1.0, help="每轮会话启动窗口秒数（0=同时启动）")
    parser.add_argument("--first-turn-file", default="", help="首轮音频文件名（留空则自动探测，如 turn_000.wav 或 turn_000_real.wav）")

    args = parser.parse_args()
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"错误：目录不存在：{audio_dir}")
        sys.exit(1)

    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    run_all_concurrency_tests(
        audio_dir=audio_dir,
        base_url=args.base_url,
        model=args.model,
        concurrency_levels=concurrency_levels,
        num_rounds=args.num_rounds,
        timeout=args.timeout,
        launch_window=args.launch_window,
        output_dir=args.output_dir,
        first_turn_file=args.first_turn_file,
    )


if __name__ == "__main__":
    main()
