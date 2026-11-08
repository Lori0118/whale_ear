# -*- coding: utf-8 -*-
"""注释账本：把「声音之外的证据」单独记账。

为什么单独记
------------
歌词、创作背景、作者经历能解释「为什么有人觉得好听」，但它们不是声学特征，
不能混进测量。混进去，测出来的就不是声音，是故事。

三条硬规则
----------
1. **设备必须记**。手机、有线麦、蓝牙耳麦的频谱倾斜差 3~6 dB/oct，环境噪声
   也各不一样。特征只在同 device + 同 env 内横向比，跨设备比分数等于比设备。
2. **verdict 和 relative 分开**。verdict 是绝对判断（好/一般/翻车），
   relative 是相对上一版的评价（比上版好）。纵向评价带「乐观偏差」，
   因为那是在跟更差的版本比；做统计只用 verdict。
3. **真人样本是基准**。`--ref` 标成参考的条目（ref3/ref4 的人声）是尺子的零点，
   合成版要跟它比，不要拿合成版互相比。

用法
----
    python notes.py add --audio out/v21.wav --device wired --env room \\
        --verdict ok --lyrics "海很深我也能唱歌" --note "咬字有雏形，音色偏薄"
    python notes.py list [--device wired] [--verdict good]
    python notes.py show <id>
    python notes.py stats
    python notes.py scan out/ --device wired --env room
    python notes.py compare          # 只输出同一 device+env 内的对比表
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "notes", "notes.json")

VERDICTS = ("good", "ok", "bad", "unrated")
RELATIVES = ("better", "same", "worse", "")
DEVICES = ("phone", "wired", "bt", "other")


def load():
    if not os.path.exists(STORE):
        return {"version": 1, "records": []}
    with open(STORE, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STORE)


def next_id(data):
    return "n%04d" % (len(data["records"]) + 1)


def cmd_add(args):
    data = load()
    rec = {
        "id": next_id(data),
        "audio": os.path.abspath(args.audio),
        "device": args.device,
        "env": args.env,
        "style": args.style,
        "ref": bool(args.ref),
        "verdict": args.verdict,
        "relative": args.relative,
        "lyrics": args.lyrics,
        "background": args.background,
        "author": args.author,
        "note": args.note,
        "tags": [t for t in (args.tags or "").split(",") if t],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data["records"].append(rec)
    save(data)
    print("已记账：%s  %s" % (rec["id"], os.path.basename(rec["audio"])))
    print("  device=%s env=%s verdict=%s relative=%s ref=%s"
          % (rec["device"], rec["env"], rec["verdict"], rec["relative"] or "-", rec["ref"]))


def cmd_list(args):
    data = load()
    rows = data["records"]
    if args.device:
        rows = [r for r in rows if r["device"] == args.device]
    if args.verdict:
        rows = [r for r in rows if r["verdict"] == args.verdict]
    if not rows:
        print("没有匹配的记录。")
        return
    for r in rows:
        flag = "R" if r["ref"] else " "
        print("%s %s %-6s %-6s %-4s %-6s %-6s %s"
              % (r["id"], flag, r["device"], r["env"], r["verdict"],
                 r["relative"] or "-", r["style"] or "-", os.path.basename(r["audio"])))


def cmd_show(args):
    data = load()
    for r in data["records"]:
        if r["id"] == args.id:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return
    print("找不到 %s" % args.id)


def cmd_stats(args):
    data = load()
    rows = data["records"]
    if not rows:
        print("账本是空的。")
        return
    print("共 %d 条" % len(rows))
    by_dev = {}
    for r in rows:
        by_dev.setdefault((r["device"], r["env"]), []).append(r)
    print("\n按设备/环境分组：")
    for (dev, env), rs in sorted(by_dev.items()):
        cnt = {}
        for r in rs:
            cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
        print("  %-6s %-10s 共 %2d 条  %s" % (dev, env, len(rs), cnt))
    print("\n提醒：不同 device 之间不要直接比分数。")
    rated = [r for r in rows if r["verdict"] != "unrated"]
    print("已判定 %d 条，未判定 %d 条。" % (len(rated), len(rows) - len(rated)))


def cmd_scan(args):
    if not os.path.isdir(args.dir):
        sys.exit("不是目录：%s" % args.dir)
    data = load()
    known = {os.path.abspath(r["audio"]) for r in data["records"]}
    added = 0
    for name in sorted(os.listdir(args.dir)):
        if not name.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
            continue
        p = os.path.abspath(os.path.join(args.dir, name))
        if p in known:
            continue
        data["records"].append({
            "id": next_id(data), "audio": p, "device": args.device, "env": args.env,
            "style": "", "ref": False, "verdict": "unrated", "relative": "",
            "lyrics": "", "background": "", "author": "", "note": "",
            "tags": [], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        added += 1
    save(data)
    print("新登记 %d 条（待判定）。用 add --verdict 或手工补 verdict。" % added)


def cmd_compare(args):
    data = load()
    rows = [r for r in data["records"] if r["verdict"] != "unrated"]
    if not rows:
        print("没有已判定的记录，无法对比。")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["device"], r["env"]), []).append(r)
    for (dev, env), rs in sorted(groups.items()):
        print("\n== %s / %s ==" % (dev, env))
        rs.sort(key=lambda x: {"good": 0, "ok": 1, "bad": 2}.get(x["verdict"], 3))
        for r in rs:
            print("  %-4s %-6s %-28s %s"
                  % (r["verdict"], r["id"], os.path.basename(r["audio"]), r["note"][:40]))


def main():
    ap = argparse.ArgumentParser(description="whale_ear 注释账本")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="新增一条记录")
    a.add_argument("--audio", required=True)
    a.add_argument("--device", choices=DEVICES, default="other")
    a.add_argument("--env", default="unknown")
    a.add_argument("--style", default="")
    a.add_argument("--verdict", choices=VERDICTS, default="unrated")
    a.add_argument("--relative", choices=RELATIVES, default="")
    a.add_argument("--lyrics", default="")
    a.add_argument("--background", default="")
    a.add_argument("--author", default="")
    a.add_argument("--note", default="")
    a.add_argument("--tags", default="")
    a.add_argument("--ref", action="store_true", help="标记为真人基准")
    a.set_defaults(func=cmd_add)

    b = sub.add_parser("list")
    b.add_argument("--device", choices=DEVICES)
    b.add_argument("--verdict", choices=VERDICTS)
    b.set_defaults(func=cmd_list)

    c = sub.add_parser("show")
    c.add_argument("id")
    c.set_defaults(func=cmd_show)

    d = sub.add_parser("stats")
    d.set_defaults(func=cmd_stats)

    e = sub.add_parser("scan")
    e.add_argument("dir")
    e.add_argument("--device", choices=DEVICES, default="other")
    e.add_argument("--env", default="unknown")
    e.set_defaults(func=cmd_scan)

    f = sub.add_parser("compare")
    f.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
