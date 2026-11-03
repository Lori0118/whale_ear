#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""whale_ear MCP server —— 把 ear / voice / harmony 包成三个工具。

传输：stdio，**换行分隔的 JSON-RPC**（MCP 规范的 stdio 传输），每行一条消息。
不依赖任何 MCP SDK，只用标准库，所以任何 Python 3.8+ 都能跑。

注册到支持 MCP 的客户端即可（例如 DSH 的 MCP 配置、Claude Desktop 的
claude_desktop_config.json）：

    {
      "mcpServers": {
        "whale_ear": {
          "command": "<python 解释器路径>",
          "args": ["D:/AI/whale_sing/whale_ear/mcp_server.py"]
        }
      }
    }

工具：
    ear_transcribe  音频 → 音符/和弦/调性/MIDI 摘要（可选分轨）
    ear_voice       「怎么唱」：颤音、谐噪比、频谱倾斜、动态、音域
    ear_harmony     和弦与调性（含调内/调外统计）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 600

TOOLS = [
    {
        "name": "ear_transcribe",
        "description": (
            "把音频转成音符事件与 MIDI/ABC 骨架，输出调性、BPM、逐小节和弦与旋律、音域。"
            "适合回答「这首歌的音高走向是什么」。"
            "注意：这是转录，乐器信息会丢失；需要配器请用分轨（stem 参数）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio": {"type": "string", "description": "音频文件绝对路径"},
                "stem": {
                    "type": "string",
                    "enum": ["vocals", "bass", "piano", "guitar", "drums"],
                    "description": "可选：先分轨，只转录这一条",
                },
                "limit_sec": {"type": "number", "description": "只分析前 N 秒"},
                "bars": {"type": "integer", "description": "每几小节汇总一次，默认 4"},
            },
            "required": ["audio"],
        },
    },
    {
        "name": "ear_voice",
        "description": (
            "测「怎么唱」：颤音速率与幅度、谐噪比（两种独立估法）、频谱倾斜、响度与动态范围、"
            "音域、峰均比。适合比较两段人声的唱法差异。"
            "注意：绝对值不可信，只能同一套方法横向比较。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"audio": {"type": "string", "description": "音频文件绝对路径"}},
            "required": ["audio"],
        },
    },
    {
        "name": "ear_harmony",
        "description": "测和弦与调性：调性、BPM、拍数、有把握的和弦拍占比、调内占比、调外和弦位置。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio": {"type": "string", "description": "音频文件绝对路径"},
                "label": {"type": "string", "description": "输出里显示的标题，可选"},
            },
            "required": ["audio"],
        },
    },
]


def run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(
            cmd, cwd=HERE, capture_output=True, timeout=TIMEOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        out = (p.stdout or b"").decode("utf-8", "replace").strip()
        err = (p.stderr or b"").decode("utf-8", "replace").strip()
        if p.returncode != 0:
            return f"[exit {p.returncode}]\n{out}\n{err}".strip()
        return out or err or "(没有输出)"
    except subprocess.TimeoutExpired:
        return f"[超时] 超过 {TIMEOUT}s，建议用 limit_sec 限制分析长度"


def call_tool(name: str, args: dict) -> str:
    audio = args.get("audio") or ""
    if not audio or not os.path.isfile(audio):
        return f"找不到音频文件: {audio!r}"

    if name == "ear_transcribe":
        cmd = [sys.executable, os.path.join(HERE, "ear.py"), audio]
        if args.get("stem"):
            cmd += ["--stem", args["stem"]]
        if args.get("limit_sec"):
            cmd += ["--limit-sec", str(args["limit_sec"])]
        if args.get("bars"):
            cmd += ["--bars", str(args["bars"])]
        return run(cmd)

    if name == "ear_voice":
        return run([sys.executable, os.path.join(HERE, "voice.py"), audio])

    if name == "ear_harmony":
        label = args.get("label") or os.path.basename(audio)
        return run([sys.executable, os.path.join(HERE, "harmony.py"), audio, label])

    return f"未知工具: {name}"


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        mid = msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "whale_ear", "version": "0.1.0"},
            }})
        elif method in ("notifications/initialized", "initialized"):
            continue
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            text = call_tool(name, args)
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": text}],
                "isError": text.startswith("[exit") or text.startswith("[超时")
                or text.startswith("找不到音频"),
            }})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
