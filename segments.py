# -*- coding: utf-8 -*-
"""段落与动机分割：把一首歌切成有意义的段，找出重复的动机，给出「亮光候选」。

边界说明（先读这条再用）
------------------------
1. 它给的是**候选段**，不是「好听」。候选分只说明这段在某几个可测维度上突出，
   不说明人听了会喜欢。判断那一步不在这里。
2. 分数只在「同一套流程 + 同一设备 + 同一环境」内横向可比。手机、有线麦、蓝牙耳麦
   的频谱倾斜差 3~6 dB/oct，跨设备比分数等于比设备。
3. 真人样本是基准。合成版要跟真人比，不要拿合成版互相比。

用法
----
    python segments.py <audio.wav> [--json] [--min-dur 3.0]

输出：段落表（起止/时长/能量/音域/动机标签/候选分）、重复动机、亮光候选 Top3。
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

HOP = 512
SR = 22050


def load_audio(path: str):
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        raise ValueError("空音频：%s" % path)
    return y, sr


def frame_features(y, sr):
    import librosa

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    k = 9
    rms_s = np.convolve(rms, np.ones(k) / k, mode="same")
    try:
        f0 = librosa.yin(y, fmin=70, fmax=900, sr=sr, hop_length=HOP)
    except Exception:
        f0 = np.full_like(rms, np.nan)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)
    return chroma, rms_s, f0, times


def find_boundaries(chroma, times, sr, min_dur):
    """用帧间音色差异的峰值找段落边界。

    为什么不用聚类：agglomerative 在整首同源的音乐上常给出交错标签，
    切出来的段又短又碎，合完只剩一段。差异曲线的峰值对「换了点什么」更敏感。
    """
    import scipy.signal as ss

    c = chroma.T.astype(float)
    n = np.linalg.norm(c, axis=1, keepdims=True)
    c = c / (n + 1e-9)
    d = 1.0 - np.sum(c[1:] * c[:-1], axis=1)
    k = max(1, int(0.5 * sr / HOP))
    d = np.convolve(d, np.ones(k) / k, mode="same")
    thr = float(np.mean(d) + np.std(d))
    gap = max(1, int(min_dur * sr / HOP))
    idx, _ = ss.find_peaks(d, height=thr, distance=gap)
    cuts = [0] + [int(i) + 1 for i in idx] + [chroma.shape[1]]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def seg_stats(chroma, rms, f0, times, s, e, rms_floor=0.0):
    t0 = float(times[min(s, len(times) - 1)])
    t1 = float(times[min(e, len(times) - 1)])
    r = rms[s:e]
    f = f0[s:e]
    if r.size:
        f = f[r >= rms_floor]
    f = f[np.isfinite(f)]
    return {
        "start": round(t0, 2),
        "end": round(t1, 2),
        "dur": round(max(0.0, t1 - t0), 2),
        "rms": float(np.mean(r)) if r.size else 0.0,
        "f0_lo": float(np.min(f)) if f.size else None,
        "f0_hi": float(np.max(f)) if f.size else None,
        "chroma": chroma[:, s:e].mean(axis=1) if e > s else np.zeros(12),
    }


def l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def find_motifs(segs):
    """同动机判定：段平均 chroma 的余弦相似度 > 0.92。"""
    vecs = [l2n(s["chroma"]) for s in segs]
    for i, s in enumerate(segs):
        s["_vec"] = vecs[i]
    tag = [None] * len(segs)
    names = "ABCDEFGH"
    nxt = 0
    for i in range(len(segs)):
        if tag[i] is not None:
            continue
        tag[i] = names[nxt % len(names)]
        for j in range(i + 1, len(segs)):
            if tag[j] is None and float(np.dot(vecs[i], vecs[j])) > 0.92:
                tag[j] = tag[i]
        nxt += 1
    for s, t in zip(segs, tag):
        s["motif"] = t
    return tag


def prominence(segs):
    rms = np.array([s["rms"] for s in segs], dtype=float)
    mean, std = float(rms.mean()), float(rms.std() or 1.0)
    for s in segs:
        z_rms = (s["rms"] - mean) / std
        if s["f0_lo"] is not None and s["f0_hi"] is not None:
            span = s["f0_hi"] - s["f0_lo"]
        else:
            span = 0.0
        z_span = span / 120.0  # 两个八度约 240Hz，取半作归一
        repeat = sum(1 for x in segs if x["motif"] == s["motif"]) - 1
        late = 1.0 if s["start"] > segs[-1]["end"] * 0.6 else 0.0
        score = 0.45 * z_rms + 0.25 * min(z_span, 2.0) + 0.15 * min(repeat, 2) + 0.15 * late
        s["cand_score"] = round(float(score), 3)
    return segs


def fmt_time(t):
    m = int(t // 60)
    s = t - m * 60
    return "%d:%04.1f" % (m, s)


def main():
    ap = argparse.ArgumentParser(description="段落/动机分割 + 亮光候选")
    ap.add_argument("audio")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--min-dur", type=float, default=6.0, help="最短段长（秒）")
    args = ap.parse_args()

    y, sr = load_audio(args.audio)
    duration = len(y) / sr
    labels_note = None
    chroma, rms, f0, times = frame_features(y, sr)
    bounds = find_boundaries(chroma, times, sr, args.min_dur)
    if not bounds:
        bounds = [(0, chroma.shape[1])]
    # 首尾残段并入相邻段
    if len(bounds) > 1:
        s, e = bounds[0]
        if times[min(e, len(times) - 1)] - times[s] < args.min_dur:
            bounds[1] = (s, bounds[1][1])
            bounds.pop(0)
        s, e = bounds[-1]
        if times[min(e, len(times) - 1)] - times[s] < args.min_dur:
            bounds[-2] = (bounds[-2][0], e)
            bounds.pop()

    floor = float(np.median(rms)) * 0.4
    segs = [seg_stats(chroma, rms, f0, times, s, e, floor) for s, e in bounds]
    find_motifs(segs)
    prominence(segs)

    for s in segs:
        s.pop("_vec", None)
        s.pop("chroma", None)
        s["rms"] = round(s["rms"], 4)
        if s["f0_lo"] is not None:
            s["f0_lo"] = round(s["f0_lo"], 1)
            s["f0_hi"] = round(s["f0_hi"], 1)

    top = [s for s in segs if s["rms"] >= float(np.median([x["rms"] for x in segs]))]
    top = sorted(top, key=lambda x: x["cand_score"], reverse=True)[:3]
    result = {
        "file": args.audio,
        "duration": round(duration, 2),
        "segments": segs,
        "top_candidates": [{"start": s["start"], "dur": s["dur"], "score": s["cand_score"]} for s in top],
        "caveat": "候选分是启发式，不是好听度；只在同设备同流程内横向可比。",
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("文件：%s  时长 %s" % (args.audio, fmt_time(duration)))
    print("-" * 78)
    print("%-4s %-16s %-7s %-9s %-6s %-6s" % ("#", "区间", "时长", "能量", "动机", "候选分"))
    for i, s in enumerate(segs, 1):
        print(
            "%-4d %-16s %-7.1f %-9.4f %-6s %-6.3f"
            % (
                i,
                "%s-%s" % (fmt_time(s["start"]), fmt_time(s["end"])),
                s["dur"],
                s["rms"],
                s["motif"],
                s["cand_score"],
            )
        )
    motifs = {}
    for s in segs:
        motifs.setdefault(s["motif"], []).append(s["start"])
    rep = {k: v for k, v in motifs.items() if len(v) > 1}
    print("-" * 78)
    if rep:
        for k, v in sorted(rep.items()):
            print("重复动机 %s：%s" % (k, "、".join(fmt_time(x) for x in v)))
    else:
        print("没有重复动机（每段 chroma 都不同）。")
    print("亮光候选：")
    for s in top:
        print("  段 %s 起 %s  得分 %.3f" % (
            segs.index(s) + 1, fmt_time(s["start"]), s["cand_score"]))
    print("\n注意：候选分是启发式，不是好听度；跨设备不可直接比。")


if __name__ == "__main__":
    main()
