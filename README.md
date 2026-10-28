# 柚子社 Galgame 角色语音合集工具

把 Galgame 解包出来的**一句一个文件**的角色语音（ogg），自动合成可以上传到视频网站（B 站等）的
**MP3 音频**和**「CG 图 + 音频」MP4 视频**，中间用本地 AI 语音识别（faster-whisper）把成人内容
挑出来剔除：先逐句打分做初筛，再对可疑句做**多轮独立识别 + 多数表决**（`--passes`，默认 3 轮），
最后还能生成一份**「疑似清单」MP3**，让你直接拖着进度条人工听一遍。

全程离线转写、不需要显卡也能跑（有 N 卡会快很多；没有 N 卡也可以按 `docs/remote_gpu.md`
租一张 4090 只跑转写，详见第 6 节）。

一次典型任务：**4000+ 句 / 4.5 小时语音**（例如 `E:\mrxyy` 的 `noz*`、`E:\etyyyyy` 的 `haz*`、
`E:\sjzmyy` 的 `nat*`、`E:\zyqhyy` 的 `nan*`），最终得到 1 个 MP3 + 1 个带封面的 MP4
（封面图由你自己提供，见 1.1）。

> 只处理你自己的、已经合法获得的游戏文件；成品仅建议作为粉丝向二创分享，不要商用。

---

## 1. 目录里都有什么

```
galgame-voice-toolkit/
├─ README.md                 ← 本文档
├─ LICENSE                   ← MIT（只覆盖代码，不含任何游戏素材）
├─ requirements.txt          ← Python 依赖（faster-whisper / numpy / Pillow）
├─ pyproject.toml            ← 包信息（pip install -e . 之后可直接用 voicepack 命令）
├─ run.bat / setup.bat       ← Windows：双击运行 / 双击装环境（装到 vendor/，免管理员）
├─ run.sh  / setup.sh        ← Linux / 算力平台：bash run.sh / bash setup.sh
├─ voicepack/                ← 处理流水线（Python 包，约 6900 行）
│  ├─ cli.py                 ← 命令行入口：python -m voicepack <子命令>（12 个子命令）
│  ├─ scan.py                ← 扫描文件夹、解析 aaa001_001.ogg 命名、排除 _ref、排队
│  ├─ asr.py                 ← AI 转写（GPU/CPU、分块、并行解码、断点续跑缓存 asr_<模型>.jsonl）
│  ├─ filter.py              ← 敏感内容加权打分 + keep/drop/review 三档 + 场景统计 + CSV 清单
│  ├─ verify.py              ← 可疑句多轮独立识别 + 多数表决（refine）
│  ├─ lexicon.py             ← 日语敏感词表 / 娇喘启发式 / 文本归一化
│  ├─ media.py               ← 母带 flac 拼接、MP3、MP4（静图）、切片、疑似清单 mp3
│  ├─ cover.py               ← 用你提供的 CG 图生成 1920×1080 封面（crop / blur）
│  ├─ remote.py              ← 远程算力：serve / worker / pull / export-jobs / import-results
│  ├─ util.py                ← 日志、控制台编码、ffmpeg 定位、纯 Python 解析 ogg 时长
│  └─ types.py               ← Clip / Decision 数据类
├─ tools/
│  ├─ qa_check.py            ← 上传前产物质检（MP4 / MP3 / 封面 / 疑似清单时间轴）
│  ├─ calibrate_filter.py    ← 抽样转写 + 阈值校准报告
│  ├─ remote_selftest.py     ← 远程算力全链路自测（本机起服务 + worker）
│  ├─ run_pipeline.py        ← 多文件夹批量：ASR 串行 + build 并发
│  ├─ run_all_folders.py     ← 多文件夹串行版本
│  └─ dev/                   ← 开发期调参/压测脚本（不用管，见 tools/dev/README.md）
├─ docs/
│  ├─ filter_notes.md        ← 过滤规则、阈值与真实数据上的实测结果
│  ├─ remote_gpu.md          ← 租 GPU 跑转写的完整教程（AutoDL / RunPod / vast.ai）
│  └─ upload_guide.md        ← 过审自检、分段、命名、驳回申诉
├─ deploy/
│  └─ remote_setup.sh        ← 云机一键装环境（Ubuntu 22.04）
└─ vendor/                   ← setup.bat 装依赖的位置（自动生成，已被 .gitignore 忽略）
```

> **仓库里不含任何游戏语音 / CG / 成品**：`.gitignore` 排除了 `vendor/`、`voicepack_out/`、`*.ogg`、
> `*.mp3`、`*.mp4`、`asr_*.jsonl` 等，仓库只有代码和文档（约 0.5MB），clone 到算力平台很快。

**语音文件夹**是你解包得到的那个目录，里面是一堆 `xxx001.ogg`、`xxx002.ogg`……（同一角色的
所有句子，文件名带场景编号）。本项目不会修改原文件夹，所有产物都写到
`<语音文件夹>\voicepack_out\`（可用 `--out` 改）。

### 1.1 封面必须由你提供 CG 图

工具**不再自动生成封面**：画面是角色 CG 图，版权和审美都该由你自己把握。

- 出 MP4 时必须给 `--image`：`python -m voicepack build "E:\mrxyy" --image "D:\cg\noz.png"`
- **没有 CG 图时只出 MP3**（不生成 MP4，也不会脑补一张渐变图充数）；
- 拿到图之后重新 `build` 一次就能补出 MP4，**不需要重新转写**（走已有的 `asr_*.jsonl` 缓存）；
- 封面比例不一定是 16:9，用 `--cover-mode` 选适配方式：

| `--cover-mode` | 效果 | 适合 |
| --- | --- | --- |
| `crop`（默认） | 居中裁剪填满 1920×1080 | 横版 CG、比例接近 16:9 的图 |
| `blur` | 整图缩放居中，四周用同一张图的**高斯模糊**填充 | 竖版/立绘，**不裁掉角色**；`--blur-strength` 调模糊半径（默认 24） |

标题区会自动排版：长标题自动缩小字号、必要时换行；`--title` / 副标题留空就不出文字。
封面统一输出 1920×1080 RGB PNG（`cover.png`），B 站投稿可以直接当封面用。

---

## 2. 安装

### 方式 A：双击 `setup.bat`（推荐）

1. 把整个项目文件夹放到一个**没有中文、没有空格的路径**下（例如 `D:\voicepack`），避免个别工具抽风。
2. 双击 `setup.bat`。它会自动：
   - 找一个 Python 3.10+（依次尝试：环境变量 `VOICEPACK_PYTHON` → 本项目 `.venv` → 系统 `python` / `py -3`）；
   - 检查 `ffmpeg` / `ffprobe`；
   - 把依赖装进本目录的 `vendor\` 文件夹（**不需要管理员权限**，也不污染系统 Python）；
   - 如果检测到 NVIDIA 显卡，会问你要不要顺带装 CUDA 运行库（约 1GB 出头，装了转写快 5–10 倍，不装也能用 CPU）。
3. 没装 FFmpeg 的话，按脚本提示二选一：
   - `winget install --id Gyan.FFmpeg -e`（推荐，免管理员）；
   - 或到 <https://www.gyan.dev/ffmpeg/builds/> 下载，解压后把 `bin` 目录加进系统 `PATH`，重开窗口。

只想检查环境、不安装：命令行执行 `setup.bat --check-only`（全部就绪返回 0）。
以前装过、想重装：`setup.bat --force`。

### 方式 B：自己用 pip 装

```bat
python -m pip install -r requirements.txt
```

再确保 `ffmpeg -version` 和 `ffprobe -version` 都能跑通。

### 方式 C：Linux / 算力平台（`setup.sh` + `run.sh`）

```bash
git clone https://github.com/<你的用户名>/galgame-voice-toolkit.git
cd galgame-voice-toolkit
bash setup.sh                # 建 .venv、装依赖，有 N 卡会自动装 cublas/cudnn
bash run.sh --help           # 看用法
bash run.sh all /data/voice/mrxyy --passes 3
```

- 装不上 ffmpeg 时：`sudo apt install -y ffmpeg`（或 `conda install -c conda-forge ffmpeg`）。
- 国内 pypi 慢：`PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash setup.sh`。
- 只想检查环境：`bash setup.sh --check-only`；不想装 CUDA：`bash setup.sh --no-cuda`。
- `run.sh` 会自动把 `vendor/`、`.deps/`、`nvidia/*/lib` 拼进 `PYTHONPATH` / `LD_LIBRARY_PATH`。
- 也可以当普通 Python 包用：`pip install -e ".[cuda]"` 之后直接 `voicepack all /data/voice/mrxyy`。

> 在**租来的 GPU 机器**上通常只需要跑 `asr`（转写），不用装 ffmpeg（解码在本地做）。
> 具体流程见第 6 节和 `docs/remote_gpu.md`。

> 首次转写会自动下载 Whisper 模型（`small` 约 480MB、`medium` 约 1.5GB、`large-v3` 约 3.1GB），
> 需要联网；下过之后会缓存在用户目录，下次不再下载。

---

## 3. 四种用法

所有命令都要**在项目根目录**（能看到 `voicepack\` 文件夹的地方）执行；
或者用 `run.bat`，它会自己切到脚本所在目录。

### 用法一：双击 `run.bat`（新手推荐）

菜单只有 5 个选项：

| 选项 | 做什么 | 什么时候用 |
| --- | --- | --- |
| `1` 一键处理 | 扫描 → 转写 → 多轮过滤 → MP3（+疑似清单）→ MP4 | 第一次跑，或者已经改好清单要重出成品 |
| `2` 只转写并生成过滤清单 | 只跑到 `filter_report.csv`（可继续跑 refine） | 想先人工检查清单再决定出不出成品 |
| `3` 只生成 MP3 | 出 MP3 + 疑似清单 mp3，不生成视频 | 只想要音频；或者还没拿到 CG 图 |
| `4` 生成视频 MP4 | 用已有转写缓存出封面 + MP4（附 MP3） | 改完清单、或者刚拿到 CG 图想换封面重出视频 |
| `5` 检查已生成的产物 | 调 `tools/qa_check.py` 体检 | 上传前最后一步 |

按提示把**语音文件夹拖进窗口**（或粘贴路径）再回车即可。中文路径没问题。
第一次选 `1` 跑完整流程；之后改完 `filter_report.csv` 再选 `4`（或 `1` / `3`），
脚本发现已有清单时会问一句清单路径，填上就按你的修改重出。

### 用法二：一条命令搞定

```bat
python -m voicepack all "E:\mrxyy" --title "四季 ナツ" --image "D:\cg\noz.png"
```

产物默认在 `E:\mrxyy\voicepack_out\`。没有 `--image` 就只出 MP3（+ 疑似清单 mp3）。

### 用法三：分步执行（推荐第一次用，方便中途人工检查）

```bat
python -m voicepack scan   "E:\mrxyy"                        :: 只统计：多少句、多长、几个场景
python -m voicepack asr    "E:\mrxyy" --model small          :: 只转写（可断点续跑）
python -m voicepack filter "E:\mrxyy"                        :: 初筛，生成 filter_report.csv / filter_summary.txt
python -m voicepack refine "E:\mrxyy" --passes 3             :: 可疑句多轮判别 → filter_report_verified.csv
::  ← 这里可以打开 filter_report_verified.csv 人工改 action 列（见第 4 节）
python -m voicepack build  "E:\mrxyy" --title "四季 ナツ" --image "D:\cg\noz.png"
```

`build` 会优先用已有的 `voicepack_out\asr_small.jsonl` 缓存，所以改完清单重跑通常只要几分钟
（重新拼音频 + 编码 MP4）。

### 用法四：只出 MP3 / 只看疑似清单 / 切片

```bat
:: 只出 MP3（不生成视频、也不需要封面图）
python -m voicepack build "E:\mrxyy" --mp3-only

:: 用 CG 图生成封面并出视频：--image 也可以不写，工具会去语音文件夹里找 cover.png / cg.png 等
python -m voicepack build "E:\mrxyy" --image "D:\cg\noz.png" --title "四季 ナツ"

:: 只重建「疑似清单 mp3」（不重新合成主音轨，几秒钟就好）
python -m voicepack suspect "E:\mrxyy"

:: 按 60 分钟切片输出（B 站单集时长有上限时用，生成 xxx_part_01.mp3/.mp4 ……）
python -m voicepack all "E:\mrxyy" --title "四季 ナツ" --image "D:\cg\noz.png" --split-minutes 60

:: 严格模式重出一版：连"待确认"的句子也一起剔除（过审更稳）
python -m voicepack build "E:\mrxyy" --strict --image "D:\cg\noz.png"
```

### 3.1 完整参数表

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `folder` | 必填 | 语音文件夹（解包后的 ogg 目录） |
| `--out DIR` | `<folder>\voicepack_out` | 输出目录 |
| `--gap 0.5` | `0.5` | 每句之间的停顿秒数 |
| `--model small` | `small` | Whisper 模型：`tiny`/`base`/`small`/`medium`/`large-v3`（越大越准越慢） |
| `--device auto` | `auto` | `auto` / `cuda` / `cpu` |
| `--workers N` | `1` | 纯 CPU 转写时的并行进程数（N 卡不用管） |
| `--no-filter` | 关 | 不过滤，全部保留（**不建议直接上传**） |
| `--strict` | 关 | "待确认"（`review`）的句子也剔除（更安全，但会误删笑声/叹气） |
| `--drop-scenes 1,2` | 空 | 整场强制剔除，支持 `12,15` 或 `noz012` |
| `--keep-scenes 3,4` | 空 | 整场强制保留（和 drop 冲突时以 drop 为准） |
| `--cascade neighbour` | `neighbour` | 上下文扩散：以 `drop` 句为种子，紧邻且分数 ≥ 2.0 的句子一并剔除；连续 ≥ 3 句分数 ≥ 2.5 的整段剔除。`scene`=整场剔除；`off`=只信逐句判定 |
| `--decisions report.csv` | 空 | 人工改过的清单，按里面的 action 覆盖判定（优先级最高） |
| `--passes N` | `3` | 可疑句多轮判别轮数（0 = 关闭，只做初筛；轮数越多越准也越慢） |
| `--second-model medium` | 空 | 多轮判别时额外用一个模型交叉验证（如 `medium` / `large-v3`，更准但更慢） |
| `--image PATH` | 空 | 角色 CG 图；不写则只出 MP3。不写路径时自动找语音文件夹里的 `cover.png`/`cover.jpg`/`cg.png`/`cg.jpg` 等 |
| `--cover-mode crop` | `crop` | 封面适配：`crop`=居中裁剪填满；`blur`=整图居中 + 模糊背景填充（竖图不裁人） |
| `--title 角色名` | 空 | 标题；不写则用"文件夹名 (场景前缀)"，也决定输出文件名 |
| `--fps 5` | `5` | 静图视频帧率（5 就够，文件小） |
| `--crf 20` | `20` | x264 质量（越小越清晰、文件越大，18–23 常用） |
| `--bitrate 192k` | `192k` | MP3 码率 |
| `--mp3-only` | 关 | 只生成 MP3，不生成封面/MP4 |
| `--no-suspect-mp3` | 关 | 不生成「疑似清单」mp3（默认会生成 `<名字>_suspect.mp3` + `.csv`） |
| `--split-minutes N` | `0` | 按 N 分钟切片；0 = 不切片 |
| `--keep-flac` | 关 | 保留中间无损母带 `tmp\master.flac`（默认完成后删除以省空间） |

查看帮助：`python -m voicepack -h`、`python -m voicepack all -h`。

子命令一览（参数并不完全相同，`-h` 看最准）：

| 子命令 | 作用 |
| --- | --- |
| `scan` | 只统计句数、时长、场景数（仅 `--out` `--gap`） |
| `asr` | 只转写（`--model` `--device` `--workers`，可断点续跑） |
| `filter` | 只做初筛 + 上下文扩散，生成 `filter_report.csv` |
| `refine` | 对可疑句多轮判别 + 多数表决，生成 `filter_report_verified.csv`（`--passes` `--second-model`） |
| `suspect` | 只根据已有清单重建「疑似清单 mp3」 |
| `build` | 用已有转写结果出 MP3（给 `--image` 才出封面 + MP4；附疑似清单） |
| `all` | 扫描 → 转写 → 多轮过滤 → MP3（+疑似清单）→ MP4，一条龙 |
| `serve` / `worker` / `pull` / `export-jobs` / `import-results` | 远程算力相关，见第 6 节 |

---

## 4. 过滤流程（重点，上传前一定要读）

转写完成后，每一句都会被打一个"敏感分"，并按分数分成三档：

| action | 含义 | 默认是否保留 |
| --- | --- | --- |
| `keep` | 保留：正常台词 | ✅ 保留 |
| `drop` | 自动剔除：分数 ≥ 3.0，明显是成人内容 | ❌ 剔除 |
| `review` | 待确认：分数 1.5–3.0，可能是喘气/暧昧台词，也可能只是叹气/笑声 | ✅ 默认保留（`--strict` 时剔除） |

**默认策略是"保守保留"**：只删高置信度的敏感句，再靠上下文扩散（`--cascade neighbour`）处理
H 场景里那种单句打分不够、连成一片的娇喘；拿不准的（`review`）先留着，由第二道防线（多轮判别）
和人工在清单里定夺。

- 上下文扩散规则：以判定为 `drop` 的句子为种子，把**紧邻**的、分数 ≥ 2.0 的句子也一并剔除；
  另外**连续 3 句以上分数 ≥ 2.5** 的整段剔除。
- `--cascade scene`：某个场景只要有一句确认剔除就**整场剔除**（最保险，但可能误删剧情段）；
  `--cascade off`：只信逐句判定，不做任何扩散。
- 整场编号也可手动干预：`--drop-scenes` / `--keep-scenes`（见 3.1 参数表）。

### 4.1 多轮判别（refine / `--passes`）：把"待确认"再筛一遍

单次识别 + 规则打分对"连续的娇喘/气声"经常只够到 `review`，不够 `drop`。多轮判别就是专门解决这个：

- 对每一句可疑句（`drop` + `review`）**独立识别 N 轮**（默认 `--passes 3`），每轮用不同的解码参数/
  段切法，再对"是否敏感"做**多数表决**——单轮偶发的误识别会被投票压下去，真敏感句则多轮都命中；
- 加 `--second-model medium` 可以再换一个模型交叉验证（更准，但耗时大约翻倍）；
- 结果写到 `filter_report_verified.csv` + `filter_summary_verified.txt`，转写缓存为
  `verify_<模型>.jsonl`（可以断点续跑，删了要重新判别）；
- 之后的 `build` / `all` 如果没显式给 `--decisions`，会**自动优先使用**
  `filter_report_verified.csv`；给了 `--decisions` 则以你改的那份为准（人工最高）；
- 关闭：`--passes 0`（只做初筛，最快）。

**耗时提示**：多轮判别只跑可疑句（本数据量级下约 1200 句）。`small` 模型 CPU 逐轮跑会比较慢
（半小时到几小时，取决于机器）；赶时间可以 `--passes 1` 或先用 `--passes 0` 出片、回头再 refine。

### 4.2 看清单 → 人工改 → 重跑

1. 先跑 `python -m voicepack filter "E:\mrxyy"`（或 `run.bat` 选 2），需要更准再跑 `refine`。
2. 打开 `E:\mrxyy\voicepack_out\filter_report_verified.csv`（没有就开 `filter_report.csv`；
   UTF-8-SIG，Excel 直接双击不乱码）。
   列固定为：`action,score,scene,idx,name,dur,text,reasons`
   - `text` 是 AI 听到的台词（日语原文），`reasons` 是命中原因（敏感词、置信度低、同场景多句命中等）；
   - **只看 `review` 和 `drop` 的行就够了**：按 `score` 降序、或按 `scene` 排一下。
3. 把 `action` 列改成你要的：`keep` / `drop` / `review` 都行，也支持中文 `保留` / `剔除` / `待确认`。
   - 不要留空；改完保存（保持 csv 格式，别存成 xlsx）。
4. 重跑时带上 `--decisions` 指定这个 csv（用 `run.bat` 选 1 / 3 / 4 时，如果发现已有清单，
   脚本会多问一句「人工修改过的清单路径」，把这份 csv 的路径填进去就行；直接回车 = 不使用）：
   ```bat
   python -m voicepack build "E:\mrxyy" --decisions "E:\mrxyy\voicepack_out\filter_report_verified.csv" --image "D:\cg\noz.png" --title "四季 ナツ"
   ```
   人工改过的 action **优先级最高**，会覆盖自动判定。

`filter_summary.txt` 是给人看的汇总：三档各多少句/多长时间、平均分、最"危险"的场景 Top5，
方便你快速判断哪些场景编号该整场砍掉（然后 `--drop-scenes`）。

### 4.3 疑似清单 MP3：直接听一遍最稳

不管过滤多准，**上传前用自己的耳朵过一遍可疑句**是最保险的。工具默认会生成：

| 文件 | 内容 |
| --- | --- |
| `<名字>_suspect.mp3` | 所有 `drop` + `review` 的句子按原顺序拼接（句间同样留 `--gap` 停顿） |
| `<名字>_suspect.csv` | 每句的时间轴：`start,end,name,action,score,votes,text,reasons`（`start/end` 就是该句在这个 mp3 里的秒数，保留 3 位小数） |

用法：用播放器打开 `<名字>_suspect.mp3`，对照 csv 拖动进度条；听到确实该删的，就把
`filter_report_verified.csv` 里对应 `name` 的 `action` 改成 `drop`（不该删的改成 `keep`），
再 `build --decisions ...` 重出即可。CSV 与 mp3 的时间轴由同一份 PCM 时长累加得出，
和 mp3 严格对齐（`tools/qa_check.py` 会校验这一点）。

- 不想要这个文件：`--no-suspect-mp3`；
- 只重建它（不重合成主音轨、几秒完成）：`python -m voicepack suspect "E:\mrxyy"`。

### 4.4 过审取向的建议

- **第一次传**建议 `--strict`：宁可少几句，也别让一条 4 小时的视频因为几分钟内容被下架。
  注意 `--strict` 是"连待确认一起删"，会误删一些无害的笑声/叹气（比如 `へへっ`），
  追求完整度的话还是按 4.2 人工复核一遍更稳。
- 如果只是"娇喘/喘息"糊成一片的段落，用 `--drop-scenes 12,15` 整场剔掉最省事。
- `--no-filter` 只适合自己留档，不要上传。

---

## 5. 产物与质检

跑完 `voicepack_out\` 里通常有：

| 文件 | 说明 |
| --- | --- |
| `<名字>.mp3` | 合成音频（48kHz 立体声，默认 192kbps） |
| `<名字>.mp4` | 静图 + 音频的视频（1920×1080 / 5fps / h264 / aac）；**只有给了 `--image` 才有** |
| `<名字>_suspect.mp3` + `.csv` | 疑似清单 mp3 与时间轴（人工试听用，见 4.3） |
| `<名字>_part_01.mp4` … | 用了 `--split-minutes` 时的分段产物 |
| `cover.png` | 封面图（1920×1080，来自你的 CG 图；没给图就没有这个文件） |
| `filter_report.csv` | 初筛清单（可人工修改后喂回 `--decisions`） |
| `filter_report_verified.csv` | 多轮判别后的最终清单（存在时 `build` 自动优先用它） |
| `filter_summary.txt` / `filter_summary_verified.txt` | 过滤统计摘要 |
| `asr_<模型>.jsonl` | 转写缓存（断点续跑用，**别删**，删了就得重转写） |
| `verify_<模型>.jsonl` | 多轮判别缓存（同上） |
| `tmp\master.flac` | 中间无损母带，仅 `--keep-flac` 时保留 |

**上传前跑一遍质检**（也可以 `run.bat` 选 5）：

```bat
python tools\qa_check.py "E:\mrxyy\voicepack_out"
python tools\qa_check.py "E:\mrxyy\voicepack_out" --expected-minutes 271
python tools\qa_check.py "E:\mrxyy\voicepack_out\noz.mp4" "E:\mrxyy\voicepack_out\noz.mp3"
```

它用 ffprobe/ffmpeg 检查：MP4 容器、h264 + yuv420p、分辨率、帧率、**moov 前置（faststart）**、
aac 48kHz 立体声、时长、MP3 与 MP4 时长是否一致、首帧和开头 5 秒能否正常解码；MP3 的采样率/码率；
封面是否 1920×1080 RGB 无透明通道；**疑似清单**的 CSV 行数与 `start/end` 时间轴是否和
`<名字>_suspect.mp3` 的时长对得上（不一致会 FAIL，说明两份文件不是一次生成的）。

另外：

- **目录里没有 MP4 不算失败**：没提供 CG 图时本来就只出 MP3（+疑似清单），qa_check 会提示一句
  「只出 MP3 也算正常」，其余检查照常 PASS；
- 输出 `[PASS]/[WARN]/[FAIL]` 和汇总，退出码 `0`=通过（可带警告）、`1`=有失败、
  `2`=没找到文件；
- `--expected-minutes 271` 用来核对总时长有没有被截断（只看主产物；`_suspect.mp3` 和 `_part_` 分段会自动跳过）；`--json` 输出机器可读结果；
  `--no-decode` 跳过解码、只做元数据检查（更快）。

---

## 6. 远程算力（本机没有好显卡时）

全量 4 个文件夹约 **23 小时、1.6 万句**，本机 CPU 跑 `large-v3` 要很久；显存不够还会 OOM。
本工具支持「**本地负责扫描/过滤/合成，租一张 4090 只跑转写**」——语音文件只在你和租来的机器之间
传输，敏感判定和词表始终留在本机，结果落回本机 `asr_<模型>.jsonl` 后，后续流程完全不变。

最短路径（本机起服务 → 远端 worker 连过来 → 拉结果）：

```bat
:: 1) 本机：开服务（只需能被远端访问到；没有公网 IP 就用 SSH 反向隧道，见 docs/remote_gpu.md）
python -m voicepack serve "E:\mrxyy" --port 8765 --host 127.0.0.1 --token <随机token>

:: 2) 租来的 GPU 机器上：跑 worker（先 --max-clips 8 小批量验证，再跑全量）
python -m voicepack worker --server http://127.0.0.1:8765 --token <随机token> --model large-v3 --device cuda

:: 3) 本机：把结果拉回来，合并进 asr_<模型>.jsonl
python -m voicepack pull --server http://127.0.0.1:8765 --token <随机token>
python -m voicepack import-results "E:\mrxyy" remote_results.jsonl

:: 4) 照常继续（用同一个 --model large-v3 才能命中缓存）
python -m voicepack filter "E:\mrxyy" --model large-v3
python -m voicepack build  "E:\mrxyy" --model large-v3 --image "D:\cg\noz.png" --title "四季 ナツ"
```

**完全没有公网入口**（公司网络限制、不想开端口）时，用离线任务包：

```bat
python -m voicepack export-jobs "E:\mrxyy"                 :: 生成 <folder>\remote_jobs\（manifest.json + ogg\）
:: 把 remote_jobs 目录打包上传到远端，在远端跑 worker --jobs-dir ... --out results.jsonl
python -m voicepack import-results "E:\mrxyy" results.jsonl
```

完整的平台选择（AutoDL / RunPod / vast.ai）、费用估算（4090 跑全量约 2~3 小时、¥5~9）、
SSH 隧道方向、装 CUDA 运行库、故障排查，见 **[docs/remote_gpu.md](docs/remote_gpu.md)**。
`suspect` / 过滤 / 合成全部在本机跑，租来的机器只做转写，跑完立刻关机。

---

## 7. 上传注意事项（B 站等）

1. **超长视频要分段**：一条几小时的视频对平台和观众都不友好，B 站单 P 也有时长/体积限制。
   用 `--split-minutes 60` 切成多段，上传时作为一个稿件的多个分 P（或独立几集），
   标题写明「P1/P2」或「1/4」。
2. **封面自己准备**：用你提供的角色 CG（`cover.png` 就是按 16:9 排好的成品），
   标题/封面/简介**不要出现**露骨词、`ASMR`/`娇喘` 之类的暗示性字眼。
3. **版权提示**：角色语音、CG 图的版权属于游戏公司（柚子社及发行商），你只是二创整理者。
   - 简介里写明「粉丝向整理，素材版权归原公司所有，侵删」，不接商业合作、不开充电专属；
   - 不要单独售卖或打包收费；
   - 如果被版权方要求下架，配合删除。
4. **过审建议**：
   - 第一次用 `--strict` 版本，并把 `<名字>_suspect.mp3` 听一遍再决定；
   - 简介加一句「内容为游戏角色台词整理，仅供学习交流」；
   - 如果只是被限流而不是下架，优先删减敏感段落后重传，别反复申诉。
5. 上传前先本地确认：`qa_check` 全部通过、用播放器把**开头 1 分钟**和**中间随机几处**点开听一下
   （确认没有漏掉的敏感句、没有静音/爆音）。

---

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| `setup.bat` 装依赖很慢/失败 | 换国内镜像重试：`python -m pip install --target vendor -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 提示没找到 FFmpeg | 按第 2 节装 FFmpeg 并把 `bin` 加进 `PATH`，**重开**命令行窗口 |
| 没有 CG 图能用吗 | 能，但只出 MP3（+疑似清单 mp3），不出 MP4、不出封面。拿到图后 `build --image 图.png` 重出视频即可（不重转写） |
| 竖版 CG 一裁剪就裁掉头 | 加 `--cover-mode blur`（整图居中 + 模糊背景填充，不裁人）；模糊太重就调 `--blur-strength` |
| 多轮判别（refine）太慢 | `--passes 1` 或先 `--passes 0` 出片；有 N 卡就确认 `--device cuda` 生效；也可以租 GPU（第 6 节） |
| 疑似清单 mp3 里听到漏删的句子 | 打开 `<名字>_suspect.csv` 找到 `name`，把 `filter_report_verified.csv` 里同名的 `action` 改成 `drop`，再 `build --decisions ...` |
| 转写特别慢 | 确认 `--device auto` 是否用上了显卡（`nvidia-smi` 能看到占用）；CPU 的话把 `--model base` 降档、`--workers` 调到 CPU 核数一半 |
| 显存不足报错 | 换 `--model small` 或 `--model base`，`--device cpu`；`large-v3` 建议租 GPU（第 6 节） |
| 报错中断了 | 直接重跑同一条命令：已转写的句子有缓存，会接着跑；refine 也有 `verify_<模型>.jsonl` 缓存 |
| 中文显示乱码 | 用 `run.bat`（已处理代码页）；命令行乱码时先执行 `chcp 65001` |
| 想换封面不重转写 | `python -m voicepack build 文件夹 --image 新图.png`（自动用缓存） |
| 视频网页上要等下载完才能拖进度条 | 说明 moov 没前置，重跑 `build`（本工具已默认 faststart），或跑 `qa_check` 看提示 |
| qa_check 提示没有 MP4 | 正常：没提供 CG 图时只出 MP3。补 `--image` 重跑 build 后再质检 |
| 磁盘空间 | 4.5 小时音频中间 flac 约 1.5GB，结束会自动删；疑似清单 mp3 每 1000 句约 100MB |
| `vendor\` 能删吗 | 能删，但下次要重新 `setup.bat` 装依赖 |

---

## 9. 免责声明

本项目只提供工具，不提供任何游戏素材。请只处理你合法拥有的游戏文件，成品仅用于个人学习和
粉丝向分享；因上传、传播产生的版权问题由使用者自行负责。
