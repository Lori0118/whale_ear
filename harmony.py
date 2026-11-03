"""按拍看和弦：chroma → 三和弦模板 → 音名 / 调内与否。只报有把握的。"""
import sys
import numpy as np
import librosa

SR = 22050
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = [0, 4, 7]
MIN = [0, 3, 7]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(chroma_mean):
    c = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    best = (-2, "")
    for i in range(12):
        for prof, mode in ((MAJOR_PROFILE, ""), (MINOR_PROFILE, "m")):
            r = float(np.dot(c, np.roll(prof / np.linalg.norm(prof), i)))
            if r > best[0]:
                best = (r, NAMES[i] + mode)
    return best[1], best[0]


def chord_of(pc_energy):
    """pc_energy: 12 维 chroma。返回 (名字, 分数)"""
    v = pc_energy / (pc_energy.max() + 1e-9)
    best = (-1, "-")
    for root in range(12):
        for tpl, suf in ((MAJ, ""), (MIN, "m")):
            t = {(root + x) % 12 for x in tpl}
            inside = float(np.mean([v[p] for p in t]))
            outside = float(np.mean([v[p] for p in range(12) if p not in t]))
            score = inside - 0.6 * outside
            if score > best[0]:
                best = (score, NAMES[root] + suf)
    return best[1], best[0]


def diatonic(chord, key):
    root = NAMES.index(chord[:-1] if chord.endswith("m") else chord)
    kval = NAMES.index(key[:-1] if key.endswith("m") else key)
    iv = (root - kval) % 12
    minor = key.endswith("m")
    if minor:
        allowed_maj = {3, 5, 8, 10}      # III V VI VII
        allowed_min = {0, 2, 7}          # i iv v
    else:
        allowed_maj = {0, 5, 7}          # I IV V
        allowed_min = {2, 4, 9}          # ii iii vi
    if chord.endswith("m"):
        return iv in allowed_min
    return iv in allowed_maj


def main(path, label):
    y, _ = librosa.load(path, sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=1024)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, units="frames", hop_length=1024)
    tempo = float(np.atleast_1d(tempo)[0])
    key, kr = key_of(chroma.mean(axis=1))
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=1024)

    seq = []
    for i, b in enumerate(bt):
        f0 = int(b * SR / 1024)
        f1 = int((bt[i + 1] if i + 1 < len(bt) else len(y) / SR) * SR / 1024)
        if f1 <= f0:
            continue
        seg = chroma[:, f0:f1].mean(axis=1)
        name, sc = chord_of(seg)
        seq.append((round(float(b), 1), name, sc, diatonic(name, key)))

    good = [s for s in seq if s[2] > 0.25]
    from collections import Counter
    cnt = Counter(s[1] for s in good)
    dia = sum(1 for s in good if s[3])
    print(f"== {label}")
    print(f"   调 {key} (r={kr:.2f})  {tempo:.0f} BPM  拍数 {len(bt)}")
    print(f"   有把握的和弦拍 {len(good)}/{len(seq)}  调内占比 {dia/max(1,len(good)):.0%}")
    print("   最常见:", ", ".join(f"{c}×{n}" for c, n in cnt.most_common(8)))
    foreign = [(t, c) for t, c, sc, d in good if not d]
    if foreign:
        # 合并相邻的同一个外音和弦
        merged = []
        for t, c in foreign:
            if merged and merged[-1][1] == c and t - merged[-1][2] <= 3:
                merged[-1] = (merged[-1][0], c, t)
            else:
                merged.append((t, c, t))
        print("   调外和弦:", ", ".join(f"{c}@{t:.0f}s" for t, c, _ in merged[:12]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
