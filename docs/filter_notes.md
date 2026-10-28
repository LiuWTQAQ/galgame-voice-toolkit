# 语音包成人内容过滤：设计、实现与校准记录

> 目标：把柚子社 Galgame 的角色语音（每文件一句 ogg）合成为长音频/视频上传视频网站之前，
> 把**露骨成人台词/娇喘**识别出来剔除或送人工确认，降低被下架的风险。
> 本文只讨论「内容审核过滤」（把已经存在的语音挑出来删掉），不涉及生成任何内容。
>
> 全部数字来自本机实测（RTX 4060 Laptop + i7，faster-whisper `small`），
> 复现命令见文末。示例文本一律截断到 20 字以内。

---

## 1. 模块与数据流

```
ogg 文件 ──scan_folder──► Clip(path,name,stem,prefix,scene,idx,dur)
                              │
                    asr.transcribe_clips
                              │  原地写 clip.text / avg_logprob / no_speech_prob / asr_done
                              ▼
                    filter.score_clip ──► (score, reasons)
                              │          lexicon.score_text（词表/短语/娇喘/幻觉）
                              ▼
                    filter.decide ──► Decision(action=keep|drop|review)
                              │          + 场景聚合、整场强制、人工覆盖、上下文扩散
                              ▼
              write_report CSV ──► 人工改 action 列 ──► read_report 回灌重跑
                               │
                    verify.verify_suspects（只对可疑句）
                               │  N 轮不同解码配置 + 多数表决
                               ▼
              filter_report_verified.csv ──► 最终 keep/drop（+ 疑似清单 mp3）
```

| 文件 | 职责 |
| --- | --- |
| `voicepack/asr.py` | ffmpeg 解码 + faster-whisper 转写；分块批量/逐句/CPU 多进程三条路径；JSONL 断点续传 |
| `voicepack/lexicon.py` | 日语敏感词加权表 + 组合短语 + 娇喘/非语言发声启发式 + 幻觉识别（纯文本，不看声学） |
| `voicepack/filter.py` | 单句打分、阈值判定、场景聚合、CSV 报告读写、中文统计摘要 |
| `voicepack/verify.py` | 可疑句多轮独立识别 + 多数表决（见第 7 节），输出最终清单与理由 |
| `tools/calibrate_filter.py` | 抽样 → ASR → 跑分 → 三桶统计/阈值扫描/疑似误杀 → 写 `stats.json` + CSV |

`voicepack/types.py` 的 `Clip` / `Decision` 与 `voicepack/scan.py` 的 `scan_folder` 直接复用，未改动。

---

## 2. ASR 侧：为什么能既快又不丢娇喘

### 2.1 不用 faster-whisper 的 PyAV 解码

本机 `av 19.0.2` 的 API 与 faster-whisper 1.2.1 不兼容（`stream.resample` 直接 AttributeError），
因此统一用 ffmpeg 子进程：

```bash
ffmpeg -v error -nostdin -i X.ogg -f f32le -ac 1 -ar 16000 -
# → np.frombuffer(stdout, np.float32)
```

单文件解码约 5ms（54x 实时），8 线程并行 + 预取后解码耗时完全被 GPU 推理掩盖。

### 2.2 分块批量转写（默认路径，GPU 首选）

按播放顺序把句子拼成 **≈240s / 最多 32 句** 的块，句间插 0.3s 静音，每块只调用一次
`model.transcribe(..., language="ja", beam_size=1, vad_filter=False,
condition_on_previous_text=False)`，然后用 segment 的绝对时间戳按
「段中点落在哪句的区间」归属回每一句（中点落在静音间隙归前一句）。

- 块不能太小：句均 5s 时 20s 一块只有 3~4 句，全量 16.7k 句要切 ~4400 块，
  每块 0.5s 固定开销就是 37 分钟。加大到 240s/块后只需 ~180 块。
- `vad_filter=False`：VAD 会把纯喘息整句吞掉（实测 `noz001_010` 开头「はぁー」被 VAD 吞），
  分块路径必须关 VAD 才能拿到娇喘文本/低置信度信号。
- 模型只加载一次（进程内 `_MODEL_CACHE`），块间复用；JSONL 缓存每句写完立即 `flush`，
  进程被杀最多丢最后一句。

实测（含模型加载）：

| 场景 | 音频 | 耗时 | RTF |
| --- | --- | --- | --- |
| GPU fp16，227s 单块（microbench） | 227s | 17~24s | 0.075~0.105 |
| GPU fp16，120 句分块（GPU 被别的任务占用时） | 556s | 92s | 0.166 |
| CPU int8 8 线程（逐句/小块） | 实测 | — | ≈0.34（2.9x 实时） |

按 0.08~0.10 估算，4 个文件夹全量 23.3h 音频的转写约 **2.0~2.5 小时**（GPU 独占时）。

### 2.3 逐句 fallback 与 VAD 吞句标记

`chunked=False` 或分块异常时退回逐句：先开 VAD，若 `segs` 为空（典型：娇喘/气声被 VAD
判成静音）自动关 VAD 重跑，并置 `clip.vad_empty = True`；filter 会把这类句子按
「疑似娇喘/气声」+1.8 分处理。分块路径无法区分「VAD 吞掉」与「本来就没段」，
但空文本同样按可疑处理（见 4.1）。

### 2.4 断点续传缓存（向后兼容）

JSONL 一行一句：`v,name,model,size,mtime,text,avg_logprob,no_speech_prob,vad_empty,dur`。
命中条件 = `name + 文件大小 + mtime(±1s) + 模型名` 全一致。
`CacheRecord.from_dict` 对缺字段全部给默认值、`v` 只记录不校验，
因此**旧版 asr.py 写的缓存（`v=1`、字段相同）可以直接读**；实测读取全量运行现场缓存
`E:\mrxyy\voicepack_out\asr_small.jsonl` 1717 行、`E:\_vp_cal\asr_small.jsonl` 400 行全部 100% 解析。
解码失败的文件不写缓存（下次自动重试），只记 `clip.asr_error`。

---

## 3. 词表与娇喘启发式（`lexicon.py`）

### 3.1 归一化

NFKC → 片假名→平假名（U+30A1–30F6 减 0x60）→ 去空白/标点/长音符 → lower。
所以「オチンチン」「おちんちんー」「おちんちん」命中同一词条，「はぁー」命中「はぁ」。

### 3.2 加权词表

| 分类 | 权重 | 例子 | 判定含义 |
| --- | --- | --- | --- |
| organ 露骨解剖 | 6.0 | おちんちん/まんこ/膣/乳首/亀頭/陰部 | 单词命中即超 drop 线 |
| act 露骨行为 | 5.0 | 中出し/挿入/ふぇら/おなにー/射精/調教 | 同上 |
| fluid 体液妊娠 | 4.0 | 精液/愛液/潮吹き/孕む | 同上 |
| body 性化身体 | 2.2 | おっぱい/おしり/えっち/すけべ/下着 | 需叠加 |
| context 性语境 | 1.5 | 気持ちいい/いっちゃう/濡れて/感じて/奥/敏感 | 单独只到 review |
| soft 暧昧词 | 0.7 | だめ/やめて/いや/いく/はぁ/そこ/ちょっと | 组合才升级 |

规则：强分类（organ/act/fluid/body）**累加**；弱分类（context/soft）**每类取最高分**，
避免「ちょっと」「そこ」这类高频日常词把普通台词刷上去。
不同弱词命中 n 个另有组合加分 `min(1.8, (n-1)×0.6)`。
`はめる→はめをはず`、`いく→いくら/いくつ`、`奥→奥さん/奥様` 等有排除窗口防误杀。

**子串去重**：同一处文本只计一次——「いやらしい」命中后不再额外计「やらしい」、
「おちんちん」不再额外计其中的短词。早期版本没去重，`noz005_024`
（「…いやらしいのはお父さんの…」）被同一处文本加了两次 2.2 分，虚高到 5.60；
去重后 2.2 分，落到 review 桶而不是 drop。

### 3.3 组合短语（单独低权重词的安全升级）

- （だめ|やめて|いや）…（いっちゃう|いくの|いかされる）+2.0
- 気持ちいい …（だめ|やめて|いっちゃう|いく）+1.5
- （吸って|舐めて|しゃぶって）…（出して|いって|だめ）+1.5（口交语境）
- （おっぱい|おしり|胸）…（大き|柔らか|気持ち|舐め|触）+1.2
- 快感词 + 同一句 ≥2 个喘息标记（あっ/はぁ/うっ/ふっ）+1.0

### 3.4 娇喘/非语言发声启发式

纯文本形态判断，必须**同时**满足「短」「无汉字」「含喘声标记」才可能是娇喘：

| 特征 | 加分 |
| --- | --- |
| 整句只由喘声字符组成（`んっ…はぁ…`） | 2.0 + 0.25×气声标记数（上限 1.0），标记≥2 再 +0.4 |
| 同字连续 ≥3（`うぅぅぅ`） | +1.0 |
| 1~2 字片段重复 ≥3 次（`あっ あっ あっ`） | +0.4+0.15×重复长度（上限 2.5） |
| 纯假名短句（≤10 字） | +0.8 |
| 气声假名占比 ≥50% | +0.6 |
| 拖长音（`ー`，≤8 字） | +0.3 |
| **笑声**（うふふ/あはは/はっはっ/へへっ…） | 压到 ≤0.5 并标注「笑声(非娇喘)降权」 |

> 早期版本用「假名占比 ≥0.75」判断娇喘，结果**所有正常日语台词**都被加 1.0 分
> （日语正常句子本来就假名占多数），review 比例虚高到 15.7%/26%。已改为上面的
> 气声标记 + 寒暄白名单（おはよう/うん/あっ/えっ 等 34 词），并对笑声单独降权。
> 修复后「おはよう」「うん」「あっ」= 0 分，「うっ、はっはっはっ!」4.4 → 0.28。

### 3.5 幻觉与退化文本

Whisper 小模型在静音/气声上会吐「ご視聴ありがとうございました」「チャンネル登録」等
YouTube 字幕幻觉，或同一 4~6 字片段重复 ≥3 次（`シシシシシシ…`）——各 +0.3，
只作提示不单独判删（这类文本既不是台词也不算露骨，但通常也不该保留语音）。

---

## 4. 单句打分与判定（`filter.py`）

### 4.1 `score_clip(clip) -> (score, reasons)`

| 项 | 分值 |
| --- | --- |
| 未转写（没跑 ASR） | 直接 review（=1.5） |
| 文本词表/短语/娇喘/幻觉 | `lexicon.score_text`（主体） |
| 空文本且 `vad_empty`（VAD 吞句） | +1.8「VAD 判定整句静音(疑似娇喘/气声)」 |
| 空文本，时长 ≥1s | +1.6「有声但转写不出，疑似喘息/喊叫」 |
| 空文本，时长 <1s | +0.4（0.3~0.9s 气口，放行） |
| `avg_logprob < -0.9 / < -1.5` | +0.8 / +1.2 |
| `no_speech_prob ≥ 0.6 / ≥ 0.85` | +0.6 / +1.0 |
| 短(≤2.5s)+低置信+高静音概率 | +0.7（喘息典型形态） |
| 极短文本+娇喘特征+高静音概率 | +0.5 |
| 同场景多句命中（场景≥2 句到 review 线、占比≥25%） | 场景内每句 +min(1.2, 0.5×(高分句数-1)) |

> 空文本为什么要 +1.6 而不是放行：实测 `haz002_001.ogg`（1.6s）transcribe 不出任何文本，
> 但 ffmpeg `volumedetect` 显示 mean −16.6dB / max −0.9dB——**音频很响**，
> 是喊叫/喘息/非语音而不是静音。对「过审」场景这类必须人工看一眼。

### 4.2 `decide()` 与优先级

```python
decide(clips, drop_threshold=3.0, review_threshold=1.5,
       drop_scenes=(), keep_scenes=(), overrides=None) -> list[Decision]
```

优先级：**单句覆盖（overrides）> drop_scenes > keep_scenes > 阈值**。
`keep_scenes -= drop_scenes`：冲突时以剔除优先（过审优先）。
`overrides` 用 stem 或文件名查，来自人工改过的 `filter_report.csv`。

### 4.3 最终阈值（实测结论）

**`drop_threshold = 3.0`、`review_threshold = 1.5`**（`filter.py` 常量默认值）。

依据（554 句真实转写缓存 + 300/100 抽样阈值扫描）：

| drop 阈值 | noz 300 句 drop/review | haz 100 句 drop/review |
| --- | --- | --- |
| 2.5 | 9 / 21 | 7 / 14 |
| **3.0** | **3 / 27** | **6 / 15** |
| 3.5 | 2 / 28 | 4 / 17 |
| 4.0 | 1 / 29 | 4 / 17 |
| 5.0 | 1 / 29 | 1 / 20 |

- 3.0~4.0 档在这 554 句里**全部**是呻吟/喊叫/成人语境（`んっ、あっ` 4.3、
  `うっ!うっ!うっ!…` 4.0、`あああああ!!` 3.6），没有一句正常台词；
  取 4.0 会把它们全留在 review 桶，人工负担反而更大。
- 2.5 会把「いやらしい」等语境词组合也判删，误伤风险开始上升，不取。
- review 线 1.5 以下样本里几乎没有可疑句（1.2~1.4 档是空文本气口和「ちょっと…」），
  再降只会灌噪音；所以 review 保持 1.5。

---

## 5. 场景聚合、上下文扩散与人工复核

### 5.1 `scene_table(decisions)`

按 `prefix + scene`（如 `noz005`）聚合：句数、keep/drop/review 数、平均分、平均时长、
总时长、剔除比例、剔除后保留时长——用于整场剔除/保留的决策（`--drop-scenes`/`--keep-scenes`）。

### 5.2 CLI 的上下文扩散（`voicepack/cli.py`，本模块提供 score）

最终 CLI 参数为 `--cascade {neighbour,scene,off}`，默认 `neighbour`：

- `neighbour`：以 drop 句为种子向左右扩散，把相邻 `score >= 2.0` 的可疑句一并剔除
  （成人场景里「一句露骨台词 + 前后几句喘息」是常态，只删中间一句经常删不干净）；
- `scene`：只要场景里有 drop，整场剔除（最保险，可能误删剧情）；
- `off`：严格按阈值，不做扩散。

### 5.3 人工复核闭环

1. 跑 `filter` 生成 `filter_report.csv`（列：`action,score,scene,idx,name,dur,text,reasons`，UTF-8-BOM）；
2. 用 Excel/WPS 打开，把 review 桶里确认过的行改成 `保留`/`剔除`（中文也行，
   `keep/drop/review`、`保留/通过/剔除/删除/待确认/复核` 都认）；
3. 重跑 `filter --decisions filter_report.csv`，`read_report` 会把 action 回灌成覆盖；
4. `--strict` 把 review 一起剔除（最保守）。

---

## 6. 校准实测

任务书要求的两次抽样（`tools/calibrate_filter.py`，seed=20241028，`small`，分块 GPU）：

### 6.1 E:\mrxyy 300 句（主校准集）

```
共 300 句 / 19分00秒，平均 3.80s
keep   270 句 (90.0%)  平均分 0.17
drop     3 句 ( 1.0%)  平均分 3.97
review  27 句 ( 9.0%)  平均分 2.05
分数分位: p50=0.0 p75=0.5 p90=1.5 p99=3.0 max=5.3
空文本 15 句 (5.0%) | 显式解剖/行为/体液词命中 0 句
高危场景: noz308(6句,drop2) noz304(8句,drop1) noz312(6句,review2) noz305(8句,review4) noz106(4句,review2)
```

### 6.2 E:\etyyyyy 100 句（交叉验证）

```
共 100 句 / 8分05秒，平均 4.85s
keep    79 句 (79.0%)  平均分 0.20
drop     6 句 ( 6.0%)  平均分 4.13   ← 全部是喘息/呻吟/成人台词
review  15 句 (15.0%)  平均分 1.77
分数分位: p90=1.9 p99=5.0 max=5.0
空文本 17 句 (17.0%) | 显式词命中 0 句
高危场景: haz416(7句,drop4,均分2.99) haz417(3句,drop1) haz405(4句,drop1)
```

etyyyyy 的 drop 率（6%）明显高于 mrxyy（1.3%），说明两个语音包成人内容密度不同；
haz416/haz417/haz405 被自动聚成高危场景，和人工听感一致。

### 6.3 全量缓存 554 句（阈值扫描集）

```
keep   470 句 (84.8%)  平均分 0.19
drop    10 句 ( 1.8%)  平均分 4.14  ← 全部为呻吟/喊叫/成人语境，无正常台词
review  74 句 (13.4%)  平均分 1.96
空文本 60 句 (10.8%，其中大部分被场景加分送 review)
```

drop 桶实际内容（截断 20 字）：

| 分数 | 文件 | 文本 | 判断 |
| --- | --- | --- | --- |
| 5.30 | noz308_246 | 「あ゛ー゛ー゛ぅぅぅぅぅ…」 | 呻吟 |
| 5.00 | haz416_190 | 「ぐう! あっ! あっ! あっ!ふっ!…」 | 呻吟+快感词 |
| 4.80 | haz417_164 | 「うっ…うっ…うっ…ウイルトの…」 | 呻吟+快感词 |
| 4.70 | haz004_024 | 「シシシシシシシシシ 知らない!…」 | 幻觉/尖叫 |
| 4.30 | haz405_032 | 「んっ、あっ」 | 纯呻吟 |
| 4.00 | haz416_275 | 「うっ!うっ!うっ!うっ!…」 | 纯呻吟 |
| 3.60 | noz308_322 | 「あああああ!!」 | 喊叫 |
| 3.40 | haz416_128 | 「悟る、あ、あ、あ、あ、悟る」 | 喘息+低置信 |
| 3.30 | haz416_229 | 「うっ…うっ…うっ…も、もう…」 | 长段喘息 |
| 3.00 | noz304_190 | 「あ、あ、あ、あ、コ、コセク…」 | 喘息 |

### 6.4 已知误杀 / 漏检（诚实记录）

- **误杀已消除的样本**：`noz005_024`（「何を考えてるのかいやらしいのはお父さんの方だよ」）
  早期因「いやらしい」+「やらしい」子串重复计分虚高到 5.60 被判删；去重后 2.2 分，落 review。
  当前 554 句 drop 桶里**没有**任何正常台词。
- **漏检风险 1：ASR 听错**。Whisper small 常把「エッチ」听成「ヘッチ」、
  把「ゴーセグ」之类专名听错，词表匹配不上 → 漏。这类只能靠 review 桶或人工复查。
- **漏检风险 2：显式词零命中**。300+100+554 句真实数据里「露骨解剖/行为/体液」词
  命中率为 **0**，词表对显式词的分值（6.0/5.0/4.0）只在合成样例上验证过
  （`中出ししちゃうよ` 5.0、`膣内に出すね` 6.0、`おっぱい舐めて` 5.5）。
  两个语音包抽样里成人内容主要表现为**喘息 + 気持ちいい 类台词**，
  不是解剖词台词；换成别的语音包时需要重新抽 100 句看显式词命中率。
- **漏检风险 3：空文本（5%~17%）**。分块路径下无法区分「VAD 吞掉」与「音频本身没人声」，
  统一 +1.6 送 review；`vad_empty` 标记只在逐句路径出现。
- **漏检风险 4：量化/设备相关的分数漂移**。校准全部用 GPU `float16` 分块（与全量生产路径一致）。
  CPU `int8` 抽查同一批成人向场景（haz416/417 抽 60 + noz309/310 共 111 = 171 句）时，
  avg_logprob 整体偏低、空文本更多（26/171 = 15%），review 比例从约 13% 涨到 34%
  （keep 112 / review 58 / drop 1）。结论：**阈值是按 GPU fp16 校准的；
  如果用 CPU int8 跑全量，review 桶会明显变大（更保守，但人工量增加），
  也可以把 avg_logprob 加分整体上移 ~0.2 再跑。**
- **笑声/尖叫边界**：`うっ、はっはっはっ!` 从 4.4 修到 0.28，但「シシシシ…」这类
  极端非语音仍会被判 drop——对「不合成长音频」这个目标来说删掉是对的。

---

## 7. 多轮独立判别（`verify.py`）

只对**可疑句**（第一轮 `review` + `drop` 桶）用不同解码配置再识别几轮，多数表决：
失败多 → 剔除，成功多 → 保留；最终清单写到 `filter_report_verified.csv`，
被剔除的句子另由 `media.py` 合成「疑似清单 mp3」供人耳复核。

### 7.1 选句与每轮配置

`plan_suspects(decisions)`：取 `action in (review, drop)` 且 `score >= REVIEW_THRESHOLD`
的句子（空文本句第一轮 1.6 分，也会入选），去重保序。

| 轮次 | 名称 | 关键参数 | 目的 |
| --- | --- | --- | --- |
| 1 | `beam5` | `beam_size=5, temperature=0.0, vad_filter=False` | 更稳的识别，减少 beam1 的随机错 |
| 2 | `vad-relaxed` | `beam_size=1, vad_filter=True, no_speech_threshold=0.95, log_prob_threshold=-2.0` | 放过 VAD，专治「娇喘被 VAD 吞掉」 |
| 3 | `beam1-nocontext` | `beam_size=1, condition_on_previous_text=False` | 独立复读，去上下文污染 |
| 4（可选） | `cross-<model>` | `--second-model medium` 时追加 | 换模型交叉验证，最贵 |

⚠️ **不要开 `temperature=[0,0.2,0.4]` 这类回退列表**：长喘息句会触发连锁温度回退，
单轮耗时从 ~100s 涨到 6 分钟以上（实测 40 句就跑了 6 分多，只能杀进程）。

轮内沿用 `asr.py` 的分块链路（`_group_chunks(240s/32 句)` + `ThreadPoolExecutor(8)`
预取解码 + 段中点归属），CPU/GPU 都支持；`device="auto"` 先试 cuda，失败回落 `cpu/int8`。

### 7.2 表决规则（保守优先）

每一轮的文本都用同一套 `filter.score_clip` 打分，标签三选一：

| 情况 | 标签 |
| --- | --- |
| 空文本 / 解码失败 | 默认 **bad**（用户要求「失败多就踢出」） |
| 分数 ≥ `REVIEW_THRESHOLD`（灰区也算） | bad |
| 有文本但整句只是喘息字符（`_moanish_only`） | bad |
| 有文本且分数 < review 线 | **good** |

**空文本什么时候弃权**：若同一句已有另一轮给出「实质正常台词」
（`_is_substantive`：归一化后 ≥8 字、含汉字、同一片段重复覆盖 <50%），
说明空文本只是这一轮 whisper/VAD 没抓到，记 **abstain** 不参与多数。
例外：第一轮（生产管线）分数已 ≥ `DROP_THRESHOLD`(3.0) 时，必须**至少两轮**
实质正常才允许弃权——防止「一轮幻觉出正常台词 + 两轮失败」放行真喘息。

**票数之外的一道闸**：若票数上该留（good > bad），但第一轮已判可疑
（`suspect_score >= 1.5`）且多轮没拿到实质正常台词，仍按剔除处理，
理由写「首轮已判可疑且多轮未恢复实质台词，保守剔除」。所以「うん」「いく!」
这类一词回答**不足以**把一句可疑句救回来。

平票剔除（过审优先）。`_moanish_only` 的字符集是
`あいうえおんっー゛ぁぃぅぇぉゃゅょゎ`（刻意不含 は/ふ/え/へ，避免把「あははは」当喘息），
长度 ≥6 且全在该集合内才算纯喘息。

理由文案示例：`多轮判别 2/3 判敏感；空文本1次`、
`多轮判别 0/3 判敏感；2 轮空文本弃权(另一轮已恢复实质台词)；转写恢复: 年末年始は忙しいから…`。

### 7.3 实测（CPU `int8`，GPU 让给全量转写）

| 样本 | 音频 | 3 轮耗时 | 结果 |
| --- | --- | --- | --- |
| 随机抽 120 句可疑句（种子 20241028，53 句首轮空文本） | 525.8s | **322s**（首跑，含模型加载；缓存命中复算 6s） | **keep 28 / drop 92** |
| 只抽空文本句 40 句（`--empty-only`） | 412.6s | 67s（前两轮命中缓存） | 40 句全 drop，无救回 |

随机 120 句的翻桶明细（第一轮 `decide()` → 多轮表决）：

| 翻桶 | 句数 |
| --- | --- |
| review → drop | 73 |
| review → keep | 27 |
| drop → drop | 19 |
| drop → keep | 1 |

空文本统计：`empty_first=64`，其中 **34 句多轮里拿到文本**（12 句判留、22 句仍删——
后者的文本本身还是喘息/短词），30 句三轮全空（无信息 → 保守剔除），
6 句最佳文本是纯喘息。**对 drop 桶几乎不翻（1/20）**，说明第一轮判删很稳；
净效果是把 review 桶里 27% 有依据地救回、73% 有依据地判掉。

全量投影（1238 句可疑 × 同分布）：keep ≈ 289、drop ≈ 949 →
最终 keep ≈ 3143/4092 = **76.8%**，drop ≈ 949 = 23.2%（不做多轮时是
keep 2854 / drop 251 / review 987，cascade 后 keep 2854 / drop 656 / review 582）。

耗时估算：1238 句 × 3 轮 = 16548s 音频。GPU `float16` 按 RTF 0.075~0.105 →
**20~30 分钟**；CPU `int8` 实测 RTF ≈ 0.61（526s 音频 322s）→ 约 2.8 小时。
`--passes 2` 可省掉最贵的 beam5 轮，代价是灰区判得糙一些。

### 7.4 已知风险（诚实记录）

- **误杀集中在「2 空 + 1 正常」票型**：120 句里 15 句判删时某轮有像正常台词的文本，
  其中约 5 句本身还是成人语境（`行く!行く!行く!`、`いいよ、だして…`、`入らないって…`），
  真正像误杀的大约 8~10 句 / 120（≈7% 可疑句，投影全量约 90 句 / 4092）。
  想更保守可把空文本全部算 bad（现状默认已经是）或直接人工过 review 桶。
- **不能靠多轮翻掉强证据**：第一轮 `>=3.0` 的句子只有两轮以上实质正常台词才可能判留，
  所以 `noz308_330`（首轮「ああああ、気持ちがよくて…」）这类会被稳定剔除，
  哪怕某一轮幻觉出「せい君はいい人だよ!」。
- **单轮幻觉**：whisper 在喘息句上会吐出正常句子（`はーい。`、`せい君はいい人だよ!`），
  所以「拿到文本」本身不足以判留，必须过 `_is_substantive` + 分数线；
  反过来它也会把正常句听成 `あっ!あっ!あっ!`（`noz304_209` 首轮 8.25 分）。
- **GPU 独占**：多轮判别是全量转写之外的第二大 GPU 消耗，别和全量转写并行
  （实测会把全量从 4.7 句/秒压到 ~1 句/秒）。校准/小样本一律 `--device cpu`。
- **缓存与断点续跑**：`verify_<model>.jsonl` 按 `(文件名, 模型, 轮次配置签名)` 缓存，
  只缓存「这一轮跑出了什么」，表决每次重算（改规则不影响缓存）。

### 7.5 CLI

```bash
# 多轮判别（读 asr_small.jsonl + filter_report.csv，写 filter_report_verified.csv）
"$PY" -m voicepack refine E:\mrxyy --out E:\mrxyy\voicepack_out
# 全流程里带上（noz 全量约 20~30 分钟 GPU；--passes 0 可关掉）
"$PY" -m voicepack all E:\mrxyy --out E:\mrxyy\voicepack_out --passes 3
"$PY" -m voicepack all E:\mrxyy --out E:\mrxyy\voicepack_out --passes 4 --second-model medium
```

产物：`filter_report_verified.csv`（最终 keep/drop，含 `votes`）、
`filter_summary_verified.txt`（中文统计）、`suspicious_*.mp3`（疑似清单，见 `media.py`）。

---

## 8. 复现命令

```bash
cd C:\Users\11424\Documents\deepseek-harness\default-workspace
PY="C:/Users/11424/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe"
export PYTHONPATH=.deps PYTHONIOENCODING=utf-8

# 1) 校准（GPU；没有 GPU 就 --device cpu，300 句约 20 分钟）
"$PY" tools/calibrate_filter.py --sample 300 --cross-sample 100
#    已有 ASR 缓存时只重跑打分（秒级，迭代阈值/词表用这个）
"$PY" tools/calibrate_filter.py --no-asr --sample 300 --cross-sample 100

# 2) 端到端（扫描→转写→过滤→MP3→MP4）
"$PY" -m voicepack all E:\mrxyy --out E:\mrxyy\voicepack_out
#    只重跑过滤（复用 asr_small.jsonl，人工改 CSV 后）
"$PY" -m voicepack filter E:\mrxyy --out E:\mrxyy\voicepack_out --decisions filter_report.csv

# 3) 只用 CPU（无 GPU / GPU 被占用）
"$PY" -m voicepack all E:\mrxyy --device cpu --workers 4
```

产物：`filter_report.csv`（逐句 action/score/reasons）、`filter_summary.txt`（中文统计）、
`.calib/stats.json`（校准确认数字）。

---

## 9. 后续可做

1. **声学特征**（不依赖 ASR）：分帧基频/能量/谱平坦度，识别「有声音但无文本」的喘息；
   目前这类靠 +1.6 送 review，若把 ffmpeg 解码后的 RMS/过零率写进缓存可选字段，
   可把「真静音」和「喘息」分开，review 桶能砍掉一半。
2. **显式词表扩充 + 真实召回验证**：换语音包时先抽 100 句看显式词命中率，
   再决定是否提高 organ/act 权重。
3. **上下文扩散默认值**：`neighbour` 的 SOFT_SCORE=2.0 是从 554 句样本目测的，
   全量跑完后按实际 review 桶再调。
