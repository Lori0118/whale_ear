#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ear.py —— 把一段音频变成我能读的东西（音符事件 / MIDI / ABC 骨架）。

链路（对应 2026-11-03 和主人定下的方案）：
    音频 → (可选) demucs 分轨 → basic-pitch(ONNX) 逐帧音符概率
         → 音符事件 → 量化到 16 分音符网格 → MIDI + ABC + 文本摘要

设计取舍（见记忆 0mw02ku4）：
  - 不做整首歌的逐音符 JSON（一首歌几万条，读起来炸上下文）；
    输出按小节聚合的 ABC 骨架 + 每 4 小节的文字摘要。
  - 保留 tempo / 拍号 / 轨名；不做「你是音乐专家」那套 prompt。
  - 目标是「判断」，不是「生成」。

用法：
    python ear.py <音频> [--stem vocals|bass|piano|guitar|drums]
                         [--bars 4] [--limit-sec 60] [--out DIR]
                         [--onset 0.5] [--note 0.3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def find_model() -> str:
    """按顺序找 basic-pitch 的 ONNX 模型：
    1) 环境变量 WHALE_EAR_MODEL 指向的文件
    2) 本目录下的 models/nmp.onnx
    3) 旧路径（本机 dsh-soundcheck 的引擎目录），仅为兼容老环境
    """
    env = os.environ.get("WHALE_EAR_MODEL")
    if env and os.path.isfile(env):
        return env
    local = os.path.join(HERE, "models", "nmp.onnx")
    if os.path.isfile(local):
        return local
    legacy = os.path.join(
        os.path.expanduser("~"),
        ".dsh", "plugins", "dsh-soundcheck", "engine", "models", "transcribe", "nmp.onnx",
    )
    if os.path.isfile(legacy):
        print("[warn] 正在使用 dsh-soundcheck 的模型，建议把 nmp.onnx 放到 ./models/",
              file=sys.stderr)
        return legacy
    raise SystemExit(
        "找不到 nmp.onnx。请下载 basic-pitch 的 ONNX 模型，放到 ./models/nmp.onnx，"
        "或用环境变量 WHALE_EAR_MODEL 指定路径。"
    )


DEFAULT_MODEL = find_model()
AUDIO_SR = 22050
WIN_SAMPLES = 43844          # basic-pitch: 1.988 s 窗口
FRAMES_PER_WIN = 172
FRAME_HOP = 256              # 22050 / 256 = 86.13 帧每秒
STRIDE_SAMPLES = 86 * FRAME_HOP  # 50% 左右重叠，且落在整帧上

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler 调性模板
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------- 加载音频
def load_audio(path: str, stem: str | None, verbose: bool,
               out_dir: str | None = None, providers: str = "cpu") -> np.ndarray:
    # 分轨一次要几分钟（CPU），存成 wav 下次直接读，别每次重跑 demucs
    cached = os.path.join(out_dir, f"stem-{stem}.wav") if (stem and out_dir) else None
    if cached and os.path.exists(cached):
        import soundfile as sf
        if verbose:
            print(f"[stem] 复用已存的分轨 {cached}", file=sys.stderr)
        y, sr = sf.read(cached, dtype="float32")
        return y if sr == AUDIO_SR else __import__("librosa").resample(y, orig_sr=sr, target_sr=AUDIO_SR)

    if stem:
        from demucs_onnx.inference import separate
        t0 = time.time()
        out = separate(path, output_dir=None, model="htdemucs", stems=[stem],
                       providers=providers, progress=False)
        y = out[stem]
        if verbose:
            print(f"[stem] demucs {stem}: {y.shape} 用时 {time.time()-t0:.1f}s",
                  file=sys.stderr)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        import librosa
        y22 = librosa.resample(y, orig_sr=44100, target_sr=AUDIO_SR)
        if cached:
            import soundfile as sf
            sf.write(cached, y22, AUDIO_SR)
        return y22

    import librosa
    y, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
    return y.astype(np.float32)


# ---------------------------------------------------------- basic-pitch ONNX
def transcribe_frames(y: np.ndarray, model_path: str, verbose: bool):
    """返回 (note[nframes,88], onset[nframes,88], contour[nframes,264])"""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    n_frames = int(np.ceil(len(y) / FRAME_HOP)) + FRAMES_PER_WIN
    acc = {n: np.zeros((n_frames, s), dtype=np.float32)
           for n, s in zip(("note", "onset"), (88, 88))}
    cnt = np.zeros(n_frames, dtype=np.float32)

    starts = list(range(0, max(1, len(y) - 1), STRIDE_SAMPLES))
    t0 = time.time()
    for k, s in enumerate(starts):
        chunk = y[s:s + WIN_SAMPLES]
        if len(chunk) < WIN_SAMPLES:
            chunk = np.pad(chunk, (0, WIN_SAMPLES - len(chunk)))
        x = chunk.reshape(1, -1, 1).astype(np.float32)
        # 注意顺序：这个导出里 :1 是 note（持续、平滑），:2 是 onset（起音尖峰），
        # :0 是 contour。别按输出名字猜，按波形形状认（2026-11-03 踩过）。
        outs = dict(zip(("onset", "note"), sess.run(out_names, {in_name: x})[:2]))
        f0 = s // FRAME_HOP
        for name in ("note", "onset"):
            arr = outs[name][0][:FRAMES_PER_WIN]
            acc[name][f0:f0 + FRAMES_PER_WIN] += arr
        cnt[f0:f0 + FRAMES_PER_WIN] += 1
        if verbose and k % 20 == 0:
            print(f"[onnx] {k+1}/{len(starts)} 窗口  {time.time()-t0:.1f}s",
                  file=sys.stderr)

    cnt = np.maximum(cnt, 1)[:, None]
    note = acc["note"] / cnt
    onset = acc["onset"] / cnt
    note = np.clip(note, 0, 1)
    return note, onset


def normalize_probs(note: np.ndarray, onset: np.ndarray):
    """这个 ONNX 是 int8 量化的，输出被压到 0.1~0.6 一带（底噪约 0.15）。
    先扣掉底噪再归一化，否则阈值不能直接用 basic-pitch 原版。
    底噪用 note 矩阵的中位数估（音乐里大部分帧本来就没这个音）。"""
    floor = float(np.median(note))
    scale = max(1e-6, 1.0 - floor)
    return (note - floor) / scale, (onset - floor) / scale


def frames_to_notes(note: np.ndarray, onset: np.ndarray,
                    onset_th: float = 0.5, note_th: float = 0.3,
                    min_frames: int = 4):
    """把概率矩阵切成音符事件。note/onset 必须已 normalize_probs。"""
    n = note.shape[0]
    sustain_th = max(0.08, note_th * 0.35)  # 持续段比起音松得多，int8 模型撑不住高阈值
    notes = []
    active = np.zeros(88, dtype=bool)
    start_f = np.zeros(88, dtype=int)
    peak = np.zeros(88, dtype=np.float32)
    last_on = np.full(88, -99, dtype=int)
    below = np.zeros(88, dtype=int)

    for f in range(n):
        row_n, row_o = note[f], onset[f]
        for k in range(88):
            # onset 上升沿且过阈 → 重击（同音重复靠这个切开）
            is_onset = row_o[k] >= onset_th and (
                (f == 0 or row_o[k] > onset[f - 1, k] + 1e-6) and
                row_o[k] >= onset[min(n - 1, f + 1), k])
            rise = row_n[k] >= note_th and (f == 0 or note[f - 1, k] < note_th)

            if active[k] and is_onset and f - start_f[k] >= min_frames:
                notes.append((21 + k, int(start_f[k]), f, float(min(1.0, peak[k]))))
                active[k] = False
            if not active[k] and (is_onset or rise) and f - last_on[k] > 2:
                active[k] = True
                start_f[k] = f
                peak[k] = row_n[k]
                last_on[k] = f
                below[k] = 0
            elif active[k]:
                peak[k] = max(peak[k], row_n[k])
                # 要连着掉下去才算断，单帧抖动不算（int8 量化后抖得很厉害）
                below[k] = below[k] + 1 if row_n[k] < sustain_th else 0
                if below[k] >= 4:
                    end_f = f - below[k] + 1
                    if end_f - start_f[k] >= min_frames:
                        notes.append((21 + k, int(start_f[k]), int(end_f),
                                      float(min(1.0, peak[k]))))
                    active[k] = False
    for k in range(88):
        if active[k] and n - start_f[k] >= min_frames:
            notes.append((21 + k, int(start_f[k]), n, float(peak[k])))

    # 去掉同音高、几乎重叠的重复（同一音被抖动切成两半）
    notes.sort(key=lambda t: (t[0], t[1]))
    merged = []
    for m, s, e, v in notes:
        if merged and merged[-1][0] == m and s - merged[-1][2] <= 2:
            pm, ps, pe, pv = merged[-1]
            merged[-1] = (pm, ps, max(pe, e), max(pv, v))
        else:
            merged.append((m, s, e, v))

    # 谐波鬼影：同一起音点上，比某个更低的音高整八度（或 12/24 半音）且明显更弱 → 丢掉。
    # 合成音色里尤其常见；真实钢琴/吉他也有。
    keep = []
    for i, (m, s, e, v) in enumerate(merged):
        ghost = False
        for j, (m2, s2, e2, v2) in enumerate(merged):
            if i == j or v2 <= v:
                continue
            if abs(s - s2) <= 3 and (m - m2) in (12, 24) and v < v2 * 0.75:
                ghost = True
                break
        if not ghost:
            keep.append((m, s, e, v))

    # loudness of vibrato: 人声一个长音会被切成 半音来回跳的一串，音符数直接翻倍。
    # 相邻（间隔≤2 帧）、音高差≤1 个半音、且后一个没有明显更强 → 并成一条。
    keep.sort(key=lambda t: (t[1], -t[0]))
    vib = []
    for m, s, e, v in keep:
        if vib:
            pm_, ps, pe, pv = vib[-1]
            if s - pe <= 2 and abs(m - pm_) <= 1 and v <= pv * 1.05:
                # 保留更长那一个的音高
                newm = pm_ if (pe - ps) >= (e - s) else m
                vib[-1] = (newm, ps, max(pe, e), max(pv, v))
                continue
        vib.append((m, s, e, v))
    return [{"midi": m, "start": s * FRAME_HOP / AUDIO_SR,
             "end": e * FRAME_HOP / AUDIO_SR,
             "vel": int(round(40 + 87 * v))} for m, s, e, v in vib]


# ------------------------------------------------------------------ 全局调性
def detect_key(y: np.ndarray):
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=AUDIO_SR).mean(axis=1)
    chroma = chroma / (np.linalg.norm(chroma) + 1e-9)
    best = (-2.0, "")
    for i in range(12):
        for prof, mode in ((MAJOR_PROFILE, "maj"), (MINOR_PROFILE, "min")):
            p = np.roll(prof / np.linalg.norm(prof), i)
            r = float(np.dot(chroma, p))
            if r > best[0]:
                best = (r, NOTE_NAMES[i] + ("" if mode == "maj" else "m"))
    return best[1], best[0]


def beat_grid(y: np.ndarray):
    import librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=AUDIO_SR, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4 or tempo <= 0:
        dur = len(y) / AUDIO_SR
        tempo = 100.0 if tempo <= 0 else tempo
        interval = 60.0 / tempo
        beats = np.arange(0, dur, interval)
    # 拍子被追到两倍（很常见）：>160 就隔一个取一个，tempo 折半，
    # 不要插中点——插了半拍会把 16 分网格挤密一倍，小节数直接翻番。
    if tempo > 160 and len(beats) > 4:
        beats = beats[::2]
        tempo = tempo / 2.0
    return tempo, beats


# -------------------------------------------------------------- 量化 & 汇总
def quantize(notes, beats, tempo, steps_per_beat=4):
    """把音符对齐到 16 分音符网格（以 beat 序列为骨架）。返回 grid notes。"""
    b = np.asarray(beats, dtype=float)
    if len(b) < 2:
        b = np.array([0.0, 60.0 / tempo])

    def to_step(t: float) -> int:
        i = int(np.searchsorted(b, t, side="right") - 1)
        i = max(0, min(i, len(b) - 1))
        if i + 1 < len(b):
            frac = (t - b[i]) / max(1e-9, b[i + 1] - b[i])
        else:
            frac = (t - b[i]) / (60.0 / tempo)
        return int(round((i + frac) * steps_per_beat))

    out = []
    for n in notes:
        s = to_step(n["start"])
        e = max(s + 1, to_step(n["end"]))
        out.append({"midi": n["midi"], "step": s, "len": e - s,
                    "vel": n["vel"]})
    # 同一 step 同一音高去重（保留更长）
    dedup = {}
    for n in out:
        key = (n["step"], n["midi"])
        if key not in dedup or dedup[key]["len"] < n["len"]:
            dedup[key] = n
    return sorted(dedup.values(), key=lambda n: (n["step"], n["midi"]))


def group_bars(gn, steps_per_bar=16):
    bars = {}
    for n in gn:
        bar = n["step"] // steps_per_bar
        bars.setdefault(bar, []).append(n)
    return bars


# ------------------------------------------------------------------- ABC 输出
def midi_to_abc(midi: int) -> str:
    """MIDI 音高 → ABC 写法。升号是前缀（^D），八度靠大小写和 ' ,。"""
    pc = midi % 12
    name = NOTE_NAMES[pc]
    acc = "^" if "#" in name else ""
    letter = name[0]
    octv = midi // 12 - 1
    if octv <= 3:
        return acc + letter + "," * (4 - octv)
    if octv <= 5:
        return acc + letter.upper() if octv == 4 else acc + letter.lower()
    return acc + letter.lower() + "'" * (octv - 5)


ABC_KEYS = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}


def abc_key(key: str) -> str:
    base, minor = (key[:-1], "m") if key.endswith("m") else (key, "")
    return ABC_KEYS.get(base, base) + minor


def abc_duration(length: int, unit=16):
    """L:1/16 下，时值为 length 个十六分音符时 ABC 的后缀。"""
    return "" if length <= 1 else str(int(length))


def build_abc(bars: dict, key: str, tempo: float, title: str, steps_per_bar=16):
    lines = ["X:1", f"T:{title}", "M:4/4", "L:1/16",
             f"Q:1/4={int(round(tempo))}", f"K:{abc_key(key)}", ""]
    max_bar = max(bars.keys()) + 1 if bars else 0
    for bar in range(max_bar):
        notes = bars.get(bar, [])
        tokens, pos, i = [], 0, 0
        while i < len(notes):
            step = notes[i]["step"] - bar * steps_per_bar
            if step >= steps_per_bar:
                break
            chord, j = [notes[i]], i + 1
            while j < len(notes) and notes[j]["step"] == notes[i]["step"]:
                chord.append(notes[j])
                j += 1
            step = max(0, step)
            if step > pos:
                tokens.append("z" + abc_duration(step - pos))
            dur = max(1, min(max(n["len"] for n in chord), steps_per_bar - step))
            names = "".join(midi_to_abc(n["midi"])
                            for n in sorted(chord, key=lambda n: n["midi"]))
            tok = "[" + names + "]" if len(chord) > 1 else names
            tokens.append(tok + abc_duration(dur))
            pos, i = step + dur, j
        if pos < steps_per_bar:
            tokens.append("z" + abc_duration(steps_per_bar - pos))
        lines.append("|" + " ".join(tokens) + "|")
    return "\n".join(lines) + "\n"


CHORD_TEMPLATES = {
    "": [0, 4, 7], "m": [0, 3, 7], "dim": [0, 3, 6],
    "aug": [0, 4, 8], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
}


def name_chord(pcs: set[int]) -> str:
    if not pcs:
        return "-"
    best = (-9.0, "-")
    for root in range(12):
        rel = {(p - root) % 12 for p in pcs}
        for suf, tpl in CHORD_TEMPLATES.items():
            t = set(tpl)
            hit = len(rel & t)
            miss = len(t - rel)
            extra = len(rel - t)
            score = hit - 0.8 * miss - 0.7 * max(0, extra - 1)
            if score > best[0]:
                best = (score, NOTE_NAMES[root] + suf)
    # 分数太差就只报音集合，别硬编和弦名
    return best[1] if best[0] >= 1.8 else "(" + "".join(NOTE_NAMES[p] for p in sorted(pcs)) + ")"


def bar_summary(bars: dict, key: str, tempo: float, steps_per_bar=16, per=4,
                with_chords: bool = True):
    """每 per 小节一行：和弦走向 + 每拍最高音。这是给我读的主视图。
    with_chords=False 时只报旋律（人声/贝斯这种单声部轨，谈和弦没意义）。"""
    out = []
    max_bar = max(bars.keys()) + 1 if bars else 0
    for start in range(0, max_bar, per):
        seg = [n for b in range(start, min(start + per, max_bar))
               for n in bars.get(b, [])]
        if not seg:
            continue
        chord_seq, melody = [], []
        for b in range(start, min(start + per, max_bar)):
            for q in range(4):
                lo, hi = b * steps_per_bar + q * 4, b * steps_per_bar + q * 4 + 4
                beat_notes = [n for n in bars.get(b, []) if lo <= n["step"] < hi]
                if not beat_notes:
                    continue
                if with_chords:
                    chord_seq.append(name_chord({n["midi"] % 12 for n in beat_notes}))
                top = max(n["midi"] for n in beat_notes)
                melody.append(f"{NOTE_NAMES[top%12]}{top//12-1}")
        lo = min(n["midi"] for n in seg)
        hi = max(n["midi"] for n in seg)
        rng = f"{NOTE_NAMES[lo%12]}{lo//12-1}–{NOTE_NAMES[hi%12]}{hi//12-1}"
        if with_chords:
            # 连续重复的和弦只留第一个
            chord_line = []
            for c in chord_seq:
                if not chord_line or chord_line[-1] != c:
                    chord_line.append(c)
            out.append(f"第{start+1}-{min(start+per, max_bar)}小节 ｜ "
                       f"和弦 {' '.join(chord_line)} ｜ 旋律 {' '.join(melody)} ｜ 音域 {rng}")
        else:
            out.append(f"第{start+1}-{min(start+per, max_bar)}小节 ｜ "
                       f"旋律 {' '.join(melody)} ｜ 音域 {rng}")
    return "\n".join(out)


# --------------------------------------------------------------------- MIDI
def write_midi(notes, tempo, path, program=0):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, n["vel"]))),
            pitch=int(n["midi"]),
            start=float(n["start"]), end=float(n["end"])))
    pm.instruments.append(inst)
    pm.write(path)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--stem", default=None,
                    choices=[None, "vocals", "bass", "piano", "guitar", "drums", "other"])
    ap.add_argument("--model", default=os.environ.get("EAR_MODEL", DEFAULT_MODEL))
    ap.add_argument("--out", default=None)
    ap.add_argument("--bars", type=int, default=4, help="摘要每几小节一行")
    ap.add_argument("--limit-sec", type=float, default=0, help="只分析前 N 秒（调试用）")
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--note", type=float, default=0.3)
    ap.add_argument("--providers", default="cpu", help="demucs 的 ORT provider：cpu / cuda / dml")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = os.path.abspath(args.audio)
    if not os.path.exists(path):
        sys.exit(f"找不到文件: {path}")
    if not os.path.exists(args.model):
        sys.exit(f"找不到模型: {args.model}")

    out_dir = args.out or os.path.join(HERE, "out",
                                       os.path.splitext(os.path.basename(path))[0]
                                       + (f"-{args.stem}" if args.stem else ""))
    os.makedirs(out_dir, exist_ok=True)

    t_all = time.time()
    y = load_audio(path, args.stem, args.verbose, out_dir, args.providers)
    if args.limit_sec:
        y = y[: int(args.limit_sec * AUDIO_SR)]
    print(f"[in] {os.path.basename(path)}  {len(y)/AUDIO_SR:.1f}s", file=sys.stderr)

    note, onset = transcribe_frames(y, args.model, args.verbose)
    note, onset = normalize_probs(note, onset)
    notes = frames_to_notes(note, onset, args.onset, args.note)
    key, kr = detect_key(y)
    tempo, beats = beat_grid(y)
    gn = quantize(notes, beats, tempo)
    bars = group_bars(gn)

    stem_name = args.stem or "mix"
    base = os.path.splitext(os.path.basename(path))[0] + (f"-{stem_name}" if args.stem else "")

    write_midi(notes, tempo, os.path.join(out_dir, base + ".mid"))
    abc = build_abc(bars, key, tempo, base)
    with open(os.path.join(out_dir, base + ".abc"), "w", encoding="utf-8") as f:
        f.write(abc)
    summary = bar_summary(bars, key, tempo, per=args.bars, with_chords=args.stem not in ("vocals", "bass"))
    tempo_note = ""
    if tempo > 130:
        tempo_note = (f"（拍点可能被追快了一倍：{tempo/2:.0f} BPM 也说得通，"
                      f"看和弦是不是两小节才换一次来判断）\n")
    with open(os.path.join(out_dir, base + "-摘要.txt"), "w", encoding="utf-8") as f:
        f.write(f"文件: {os.path.basename(path)}\n轨: {stem_name}\n"
                f"调性: {key}(相关度 {kr:.2f})  速度: {tempo:.1f} BPM\n"
                f"{tempo_note}"
                f"音符: {len(notes)} 个，量化后 {len(gn)} 个\n\n{summary}\n")
    with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(path), "stem": stem_name,
                   "key": key, "key_r": round(kr, 3),
                   "tempo": round(tempo, 2), "n_notes": len(notes),
                   "notes": notes[:2000]}, f, ensure_ascii=False, indent=1)

    print(f"[out] {out_dir}")
    print(f"      调性 {key} (r={kr:.2f})  {tempo:.0f} BPM  音符 {len(notes)} 个")
    print(f"      用时 {time.time()-t_all:.1f}s")
    print("\n" + summary[:2500])


if __name__ == "__main__":
    main()
