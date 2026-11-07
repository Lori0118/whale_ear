# whale_ear —— 给「只有文本能力」的 AI 装一副耳朵

**当前版本：v1.0**（2026-11-08 随第二期视频公开）。版本号规则：小的阶段性更新 +0.1，整体性大改整数位 +1，见 `CHANGELOG.md`。

把一段音频变成**能读、能判断**的结构化描述：音符、和弦、调性、结构、人声的唱法。

它是 [dsh-soundcheck] 的思路延伸——那套工具回答「这首歌听起来是什么样」，
whale_ear 多走一步：把歌变成**可以逐条质证的数据**，让模型能自己举证，而不是依赖别人的形容词。

> 名字：whale_ear。所属项目 whale_sing（让鲸鱼娘学会自己唱歌）。
> 第一期只做「听」。《让鲸鱼娘学会自己唱歌 01》就是这个仓库的产物。

---

## 它做什么

三个入口，各管一件事：

| 文件 | 作用 | 输出 |
|---|---|---|
| `ear.py` | 音频 → 音符事件 → 量化 → MIDI / ABC / 文本摘要 | `.mid` `.abc` `.json` `-摘要.txt` |
| `harmony.py` | 和弦、调性、段落色彩 | 文本 |
| `voice.py` | 「怎么唱」：颤音速率与幅度、谐噪比、频谱倾斜、动态范围、音域、峰均比 | 文本 |
| `mcp_server.py` | 把上面三个包成 MCP 工具，给智能体直接调用 | JSON-RPC |

还有一份 `SKILL.md`：**教智能体怎么读这些数字**——哪几项可以信、哪几项容易误判、同一个结论至少要两个独立来源。工具给数据，这份文件管判读。

设计上刻意不做两件事：
- **不输出整首歌的逐音符 JSON**（一首歌几万条，模型读不动）；按小节聚合成 ABC 骨架 + 每 4 小节一段文字摘要。
- **不生成音乐**。它只负责「判断」，不负责「创作」。

---

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 模型

需要 basic-pitch 的 ONNX 模型 `nmp.onnx`（Spotify basic-pitch 一系）。
模型已经放在仓库里（`models/nmp.onnx`，约 225 KB），克隆下来直接用，不需要额外下载。
如需换成别的版本，用环境变量指定：

```bash
export WHALE_EAR_MODEL=/path/to/nmp.onnx
```

找不到模型时会直接报错并提示，不会静默降级。

分轨（`--stem`）依赖 `demucs-onnx`（不在默认依赖里，按需装）：

```bash
pip install demucs-onnx
```

首次运行会自己下载 htdemucs 权重。

---

## 先花两分钟确认它能跑

```bash
python ear.py examples/test-chords.wav      # 应该出 out/test-chords/ 下的 mid + abc + 摘要
python harmony.py examples/test-happybirthday.wav
python voice.py examples/test-happybirthday.wav
```

三条都能出结果，环境就对了。若在 `write_midi` 处报 `ModuleNotFoundError: pretty_midi`，
说明 `pip install -r requirements.txt` 没跑全。

---

## 给智能体用：MCP

`mcp_server.py` 用标准库手写换行分隔的 JSON-RPC，不依赖 `mcp` SDK，Python 3.8+ 就能跑。
注册三个工具：`ear_transcribe` / `ear_voice` / `ear_harmony`，内部调对应脚本并回传文本。

```bash
python mcp_server.py        # 走 stdio，交给宿主去握手
```

接上之后，模型可以直接要「这首歌的和弦走向」，而不是让人把输出复制进来。
但**别指望它自己会判读**——判读规则在 `SKILL.md`，两边要一起给。

---

## 用法

```bash
# 整首歌：出 MIDI + ABC + 摘要
python ear.py examples/test-chords.wav

# 只要人声那条轨（先分轨再转录）
python ear.py song.mp3 --stem vocals

# 只分析前 60 秒，每 8 小节汇总一次
python ear.py song.mp3 --limit-sec 60 --bars 8

# 「怎么唱」
python voice.py outputs/song-vocals.wav

# 和弦与调性
python harmony.py examples/test-happybirthday.wav
```

输出默认写到 `./out/<音频名>/`。

---

## 已知限制（请先读这段）

1. **它是测量，不是听感。** 它能报「这里响了 6 dB」，报不了「这里好听」。所有「感觉」的结论都是使用者自己加的，加错了不怪工具。
2. **乐器识别不准。** 声源标签来自西方流行乐训练的模型，中文乐器经常报错（实测把唢呐报进过没有唢呐的歌）。它给的乐器名只能当方向。
3. **分轨有串音。** 人声占比、谐噪比这类数字只能同流程横向比较，不能当成「这个人气声重」的绝对判断。
4. **和弦只到根音级。** 七度、挂留、转位在混音里分不干净。
5. **中文歌词它不认。** 转录是音符级的，不转写歌词。

---

## 许可

本仓库代码：**Apache License 2.0**，全文见 `LICENSE`，第三方组件与模型的来源见 `NOTICE`。

**模型**：`models/nmp.onnx` 是 [basic-pitch](https://github.com/spotify/basic-pitch)（Spotify，Apache-2.0）的 ONNX 量化版，
由 ALittleLov（[dsh-soundcheck](https://github.com/ALittleLov/dsh-soundcheck)）转换。按 Apache-2.0 的要求，本仓库随附了协议全文与来源说明；
该文件相对原模型的改动（TF → ONNX、int8 量化）已在 `NOTICE` 中标明。

**分轨权重**不分发：`demucs-onnx` 在首次运行时自行下载 htdemucs 权重，受其上游许可约束。

---

## 致谢

- [basic-pitch](https://github.com/spotify/basic-pitch)（Spotify）——音符转录模型
- [dsh-soundcheck](https://github.com/ALittleLov/dsh-soundcheck)（ALittleLov）——本项目思路的起点，以及模型的 ONNX/int8 转换；
  「让 AI 老实说自己听不见什么」这条原则，是这个项目最好的一部分

