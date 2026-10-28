# 远程 GPU 转写：把 ASR 放到租来的显卡上跑

> 场景：全量 4 个语音文件夹合计 **23.3 小时、约 1.67 万句**，本机是 RTX 4060 Laptop（8GB），
> 而且显存经常被桌面程序/别的任务占到 7GB，跑 `large-v3` 要么 OOM 要么排队。
> 解法：**本地负责扫描/过滤/合成（数据不出门、结果落本地），租一张 4090/A100 只负责
> faster-whisper 推理**，按小时计费，跑完立刻关机。本文给出 AutoDL / RunPod / vast.ai
> 三家的完整操作步骤、组网方式、费用估算和故障排查。
>
> 所有价格与耗时都是 **2025 年的经验值/估算**（按 23.3h 音频、`large-v3`、fp16 计算），
> 平台调价、显卡型号、网络、块大小都会影响结果，实际以账单为准，误差按 ±40% 看。
> 示例文本一律截断，不展示露骨台词。

---

## 目录

1. [为什么值得租卡](#1-为什么值得租卡)
2. [架构与数据流](#2-架构与数据流)
3. [费用与耗时估算](#3-费用与耗时估算)
4. [三家平台怎么选](#4-三家平台怎么选)
5. [组网方式（关键）](#5-组网方式关键)
6. [完整流程：AutoDL 为例](#6-完整流程autodl-为例)
7. [RunPod / vast.ai 要点](#7-runpod--vastai-要点)
8. [备选：离线任务包（完全没有公网入口）](#8-备选离线任务包完全没有公网入口)
9. [命令速查](#9-命令速查)
10. [故障排查](#10-故障排查)
11. [安全与已知限制](#11-安全与已知限制)

---

## 1. 为什么值得租卡

本机实测（`docs/filter_notes.md`，faster-whisper `small`）：GPU fp16 RTF ≈ 0.075~0.166，
CPU int8 8 线程 RTF ≈ 0.34。换成 `large-v3` 参数约 6 倍于 `small`、同卡大约慢 3~4 倍，
本机 4060 Laptop 还被占用，全量 23.3h 要跑很久。

租 4090 时 `large-v3` 的 RTF 大约 0.08~0.12（**估算**，见第 3 节），23.3h 音频 **2~3 小时**
能跑完，费用不到 10 块钱。而本地可以同时继续做扫描、过滤、封面、MP3/MP4 合成。

收益不只是快：

- 显存 24GB/40GB，`large-v3` + `float16` 不用缩块，识别质量比 `small` 稳；
- 本机 GPU 空出来给别的任务，不会被 ASR 长时间占住；
- 结果依然**全部落在本机**的 `asr_<model>.jsonl`，后续 `filter/build` 流程零改动。

---

## 2. 架构与数据流

```
本机（扫描/过滤/合成）                                租来的 GPU（只做推理）
──────────────────────────────────                  ─────────────────────────
scan_folder(folder) → [Clip...]
RemoteServer（voicepack/remote.py）
  · done 依据 = remote_inbox.jsonl
  · 派发带约 4 分钟在途租约（worker 用 ?claim=1 领取）
        │
        │  GET /api/jobs?limit=32   ───────────────►  run_worker()
        │  GET /api/file/<name>     ───────────────►  下载 ogg 到临时目录
        │                                              asr.transcribe_clips()
        │                                              （240s/32 句一块，复用 asr.py）
        │  POST /api/result         ◄──────────────  逐条回传 text/avg_logprob/…
        ▼
  remote_inbox.jsonl（追加写，断点续跑）
pull_results()  →  remote_results.jsonl
import_results() →  <folder>/voicepack_out/asr_large-v3.jsonl（与 AsrCache 兼容）
之后照常：python -m voicepack filter / build
```

关键点：

- **服务端是唯一的任务队列**：`/api/jobs` 只返回「不在缓存里且不在在途租约里」的任务，
  worker 挂了重启会接着拉；GPU 机器被回收了也不丢进度（每句回传后立即写入）。
- **worker 不做过滤**：它只做转写，文本原样回传；敏感内容判定全部留在本机
  （`filter.py`/`lexicon.py`），避免把判断逻辑和词表传到租来的机器上。
- **结果缓存格式与本地完全一致**：`import` 后就是 `asr_<model>.jsonl`，
  `AsrCache.get_valid` 按 `name + size + mtime(±1s) + model` 命中，不会重转写。
- **可以随时 pull**：worker 还在跑的时候也能 `pull` 已经回传的部分，不用等全部结束。

---

## 3. 费用与耗时估算

**假设**（估算基准）：4 个文件夹共 23.3h 音频（23.3 × 3600 = 83,880 秒）、约 16.7k 句、
句均 5s、块 240s/32 句、`large-v3`、`float16`、beam_size=1、`vad_filter=False`
（和本地 `asr.py` 默认一致）。平台价格按 2025 年公开报价取区间，**不含**存储、
公网流量、镜像拉取费用。

| 平台 | 显卡 | RTF（估算） | 纯转写耗时 | 单价（估算） | 总费用（估算） |
| --- | --- | --- | --- | --- | --- |
| AutoDL | RTX 4090 24G | 0.08~0.12 | 2.0~3.0 h | ¥2.0~2.5/h | **¥5~9** |
| AutoDL | A100 40G | 0.06~0.09 | 1.5~2.3 h | ¥5.5~7.0/h | **¥11~18** |
| AutoDL | A100 80G | 0.05~0.08 | 1.2~1.9 h | ¥8~10/h | **¥12~22** |
| RunPod | RTX 4090 | 0.08~0.12 | 2.0~3.0 h | $0.34~0.69/h | **$1.0~2.5** |
| RunPod | A100 80G | 0.05~0.08 | 1.2~1.9 h | $1.64~2.49/h | **$2.5~5.5** |
| vast.ai | RTX 4090 | 0.08~0.12 | 2.0~3.0 h | $0.30~0.50/h | **$0.9~1.8** |
| vast.ai | A100 40G/80G | 0.05~0.09 | 1.2~2.0 h | $0.80~1.80/h | **$1.5~4.0** |

RTF = 转写耗时 / 音频时长，越小越快。估算依据：本机 4060 Laptop `small` fp16 实测
RTF 0.075~0.166；`large-v3` 约为 `small` 的 3~4 倍耗时；4090 约为 4060 Laptop 的 2.5~3.5 倍；
A100 80G 显存带宽 2TB/s（4090 约 1TB/s），再快 1.5~2 倍。实际会被 CPU 解码、块大小、
beam、显卡共享/降频影响。

**别忘了加的准备时间**（按小时计费，建议整体留 30~60 分钟 buffer）：

| 步骤 | 时间 |
| --- | --- |
| 创建实例 + 拉镜像 + 登录 | 3~10 分钟 |
| 上传项目代码（<1MB，或 git clone） | 1~2 分钟 |
| `deploy/remote_setup.sh` 装依赖 | 3~8 分钟 |
| 下载 `large-v3` 模型（~3.1GB，国内建议走镜像） | 2~10 分钟 |
| 跑 worker（按上表） | 1.2~3 小时 |
| 下载结果 JSONL（16.7k 行，约 3~5MB） | 几秒 |
| 关机/销毁实例 | 1 分钟 |

**传输量**：

- 在线模式：本机只上传「待转写 ogg」，4129 个文件约 **212MB**（`E:\mrxyy` 实测）；
- 离线任务包：同样约 212MB，`zip` 后体积变化不大（ogg 已压缩），但**文件数从 4129 变 1**，
  上传网盘/断点续传会稳很多；
- 结果回传：16.7k 行 JSONL 约 3~5MB。

结论：**4090 就够，2~3 小时、¥5~9（或 $1~2）**；A100 更快但性价比一般，只有赶时间才选。

---

## 4. 三家平台怎么选

| 维度 | AutoDL（推荐新手/国内） | RunPod（推荐海外/信用卡） | vast.ai（最便宜） |
| --- | --- | --- | --- |
| 支付 | 支付宝/微信，需实名 | 信用卡/PayPal | 信用卡/PayPal |
| 计费 | 按量，关机只收存储 | **按秒** | **按秒**，竞价实例更便宜 |
| 4090 供给 | 紧张，常需抢 | 充足 | 充足（主机质量参差） |
| 网络 | SSH 端口直连，国内访问快 | SSH over TCP，需挂 SSH key | SSH 直连，需挂 SSH key |
| 公网端口 | 不提供（正好用隧道） | 端口只暴露 pod 内服务 | 实例端口只暴露实例内服务 |
| 上手难度 | 低（中文界面、镜像多） | 中 | 中（竞价实例可能被回收） |
| 适合 | 国内用户、第一次用 | 有信用卡、想要按秒计费 | 追求最低价、能接受被回收 |

**推荐**：

- 第一次用、要中文、能支付宝 → **AutoDL RTX 4090**（`¥2.0~2.5/h`，2~3 小时跑完）；
- 海外信用卡、想按秒计费 + 随时停 → **RunPod 4090**（community cloud `$0.34/h` 起）；
- 极限省钱、能接受机器被抢 → **vast.ai 4090 竞价**（`$0.30/h` 左右，但要有重跑的心理准备）。

三家都建议：**先开 1 小时做 smoke test（先跑 8 句小批量），确认环境通了再整机跑全量**。

---

## 5. 组网方式（关键）

方向永远是：**本机起 `serve`，远端的 worker 主动连本机**。所以问题是：远端怎么访问到本机？

### 5.1 方式 A：SSH 反向隧道（**推荐**，本机没公网 IP 也能用）

在**本机**执行（Windows PowerShell / Git Bash / WSL 都行）：

```powershell
# 1) 本机启动服务（只监听本机回环，不需要暴露公网）
python -m voicepack serve "E:\mrxyy" --model large-v3 --port 8765 --host 127.0.0.1 --token <随机token>

# 2) 另开一个窗口，建立反向隧道：远端 127.0.0.1:8765 → 本机 127.0.0.1:8765
ssh -p <远端SSH端口> -R 8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 root@<远端主机> -N
```

然后在**远端**跑 worker，直接连 `127.0.0.1`：

```bash
python -m voicepack worker --server http://127.0.0.1:8765 --token <随机token> \
    --model large-v3 --device cuda
```

> ⚠️ **方向别搞反**：这里用的是 `-R`（remote forward，让远端监听 8765 再转发回本机）。
> 常见写法 `ssh -L 8765:127.0.0.1:8765` 是 local forward（让**本机**监听 8765 转发到远端），
> 方向正好相反，用在「本机 serve + 远端 worker」这个场景会连不上。

隧道排查：

- 报 `remote port forwarding failed for listen port 8765` → 远端 8765 已被占用，换端口：
  `-R 18765:127.0.0.1:8765`，worker 用 `--server http://127.0.0.1:18765`；
- 一直断 → 加 `-o ServerAliveInterval=30 -o ServerAliveCountMax=3`，或者用
  `autossh -M 0 -R ...`；
- 远端 sshd 若配置了 `AllowTcpForwarding no`（极少见）则隧道不可用，改用方式 C。

### 5.2 方式 B：公网 / 局域网直连

适用于：本机有公网 IP（或云服务器端口映射）、或者远端与本机在同一局域网。

```powershell
# 本机：监听所有网卡，必须设 token
python -m voicepack serve "E:\mrxyy" --model large-v3 --port 8765 --host 0.0.0.0 --token <强随机token>
# 远端：
python -m voicepack worker --server http://<本机公网或局域网IP>:8765 --token <强随机token> --model large-v3 --device cuda
```

注意：

- **必须设置 token**（`serve` 不带 `--token` 会随机生成并打印，不会裸奔；但公网直连请手动设长 token）；
- 路由器/云防火墙只临时放行该端口，最好限制来源 IP 为租用平台的出口 IP（较难确定，做不到就
  跑完立刻关）；
- 明文 HTTP，token 会在网络上传输，**不要在公共 WiFi 下裸奔**；能用隧道就用隧道；
- Windows 防火墙需要放行 Python 的入站连接（第一次运行会弹窗，选「专用网络」）。

### 5.3 方式 C：离线任务包（没有公网入口 / 不想开端口）

见第 8 节，完全不建连：本机导出任务包 → 上传网盘/OSS → 远端下载 → 跑 → 下载结果 → 导入。

---

## 6. 完整流程：AutoDL 为例

### 6.1 租机器

1. 注册/登录 <https://www.autodl.com>，充值（学生认证有优惠）；
2. 「算力市场」里选 **RTX 4090 / 24GB**（或 A100），镜像选
   **PyTorch 2.x + CUDA 12.x**（Ubuntu 22.04），计费方式选**按量计费**；
3. 创建实例，等状态变成「运行中」；
4. 实例页复制 SSH 登录命令，形如：

   ```bash
   ssh -p 32222 root@connect.westb.seetacloud.com
   ```

   密码在页面里（或已配置好公钥）。

### 6.2 上传项目 + 装环境（远端）

worker 需要 `voicepack/` 包和 `deploy/remote_setup.sh`。两种方式二选一：

```bash
# 方式 1：从本机 scp 上传（本机执行；把 voicepack/ deploy/ requirements.txt 传上去）
scp -P 32222 -r voicepack deploy requirements.txt root@connect.westb.seetacloud.com:/root/vp/

# 方式 2：远端 git clone 你自己的仓库（AutoDL 可先 source /etc/network_turbo 加速）
git clone <你的仓库地址> /root/vp
```

装环境（远端执行，国内建议挂镜像）：

```bash
cd /root/vp
export HF_ENDPOINT=https://hf-mirror.com          # 模型下载走镜像，别省
export PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
bash deploy/remote_setup.sh /root/vp
```

脚本会：建 `.venv-remote/`、装 `faster-whisper`、检测到 `nvidia-smi` 时补装
`nvidia-cublas-cu12` + `nvidia-cudnn-cu12==9.*`（ctranslate2 4.x 在 CUDA 12 下必需的库），
最后用 1 秒静音 wav 做 smoke test（CPU int8 + 有 GPU 时再验证 CUDA float16）。
看到 `安装完成 ✅` 和 `CUDA float16 加载 + 转写 OK` 就说明环境没问题。

### 6.3 本机起 serve + 建隧道

本机（项目根目录，先 `cd` 到项目里）：

```powershell
set PYTHONPATH=.deps
python -m voicepack serve "E:\mrxyy" --model large-v3 --port 8765 --host 127.0.0.1 --token my-secret-2025
```

`python -m voicepack serve` 会把结果直接写进 `<folder>\voicepack_out\asr_large-v3.jsonl`
（就是本地 `asr` 用的缓存文件，`--model` 决定文件名）；**重复运行会自动续跑**
（已经转写过的句子不会再派发）。按 `Ctrl+C` 停止。

本机另开一个窗口建反向隧道：

```powershell
ssh -p 32222 -R 8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    root@connect.westb.seetacloud.com -N
```

### 6.4 远端跑 worker

```bash
cd /root/vp
source .venv-remote/bin/activate      # 或直接用 .venv-remote/bin/python
export HF_ENDPOINT=https://hf-mirror.com
python -m voicepack worker --server http://127.0.0.1:8765 --token my-secret-2025 \
    --model large-v3 --device cuda \
    --out-jsonl /root/results.jsonl
```

看到这样的进度就说明通了：

```
[remote] 服务端 http://127.0.0.1:8765：共 4129 句，已完成 0，待转写 4129
[remote] 已完成 32/4129 (0.8%) 0.4 句/秒  剩余约 160.0 分钟
[remote] 已完成 64/4129 (1.6%) 0.5 句/秒  剩余约 130.0 分钟
...
```

> 建议先小批量试一下环境：加 `--max-clips 8` 跑一小会儿，看文本质量/RTF 正常再全量
> （`--max-clips` 同时是「每批任务数」和「每块句数」，正式跑用默认 32）。
>
> `voicepack worker` 没有 `--compute-type` 参数：`--device cuda` 时内部自动 `float16`、
> `--device cpu` 时自动 `int8`（和 `voicepack asr` 一致）。要显式指定量化（如 `int8_float16`）
> 就用模块命令行：`python -m voicepack.remote worker ... --compute-type int8_float16`。

worker 可以随时 `Ctrl+C` 停，重新启动会接着拉没做完的（服务端缓存 + 约 4 分钟租约兜底）。
如果 worker 崩溃，最多约 4 分钟后那些在途任务会自动重新派发给下一个 worker。

### 6.5 拉结果 + 合并到本地

worker 结束（或随时）在本机执行：

```powershell
# 下载服务端累积的全部结果（覆盖写，原子替换）
python -m voicepack pull --server http://127.0.0.1:8765 --token my-secret-2025 `
    --out-jsonl E:\mrxyy\voicepack_out\remote_results.jsonl

# 按模型分组合并进 asr_<model>.jsonl（与 AsrCache 完全兼容，按 name 去重、原子写）
python -m voicepack import-results "E:\mrxyy" E:\mrxyy\voicepack_out\remote_results.jsonl
```

> 子命令名是 `import-results`（`import` 在命令行里容易和 Python 语义混淆，故加了后缀），
> 参数顺序是 `<语音文件夹> <results.jsonl>`；默认写进 `<folder>\voicepack_out\`，用 `--out` 可改。
> 等价写法：`python -m voicepack.remote import <results.jsonl> --cache-dir <folder>\voicepack_out`。
> 注意 `--model` 要与之后本地命令用的模型名一致（默认 `large-v3` → `asr_large-v3.jsonl`）。

之后就是原来的本地流程，完全不用改：

```powershell
python -m voicepack filter "E:\mrxyy" --model large-v3
python -m voicepack build  "E:\mrxyy" --model large-v3 --title 野宫
```

### 6.6 关机

跑完立刻在 AutoDL 控制台**关机**（按量计费关机后只收存储费；「释放」则清空磁盘）。
隧道窗口 `Ctrl+C`，本机 serve `Ctrl+C`。结果 JSONL 已经落在本机，远端可以随便清。

---

## 7. RunPod / vast.ai 要点

### 7.1 RunPod

1. 注册 <https://www.runpod.io>，Add credit，在 Settings → SSH Public Keys 里填自己的公钥
   （`~/.ssh/id_ed25519.pub`）；
2. Deploy Pod → 选 GPU（RTX 4090 / A100）→ 模板选
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`（或任意带 CUDA 12 的 PyTorch 镜像），
   容器盘 20GB、按秒计费；
3. Pod 详情页 → Connect → **SSH over exposed TCP**，会给出形如
   `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519` 的命令；
4. 上传项目（`scp -P <port> -r voicepack deploy root@<ip>:/root/vp/`），装环境同 6.2；
5. 反向隧道（**本机执行**，端口用 RunPod 给的 SSH TCP 端口）：

   ```bash
   ssh -p <port> root@<ip> -R 8765:127.0.0.1:8765 -N
   ```

   然后远端 worker 连 `http://127.0.0.1:8765`。

注意：RunPod 的「HTTP Ports」是给 **pod 内服务** 暴露公网用的，和我们要的
「pod 主动连本机」无关，别配错方向。

### 7.2 vast.ai

1. 注册（GitHub 登录）→ 充值 → 在 Account 里加 SSH 公钥；
2. Search 里筛选 `RTX 4090` / `A100`，机器选 **On-demand**（稳）或 **Interruptible**（便宜，
   可能被回收，worker 支持断点续跑所以问题不大），镜像选
   `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` 或 `nvidia/cuda:12.4.1-devel-ubuntu22.04`，
   disk 20GB；
3. 创建实例，实例页给 `ssh -p <port> root@<host>`；上传项目、装环境、建隧道完全同 6.2/6.3；
4. 用完 **Destroy**（按秒计费，不销毁会一直扣）。

> 三家通用的坑：镜像里的 CUDA 版本要和 `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`
> 匹配（CUDA 12 + cuDNN 9）。如果 `remote_setup.sh` 的 smoke test 在 CUDA 那步报
> `Could not locate cudnn_ops.so`，就是这套库没装对，重新跑一遍脚本即可。

---

## 8. 备选：离线任务包（完全没有公网入口）

适合：本机没有公网 IP 又不想开隧道、公司网络限制出站、或者只想租一次机器不开着电脑等。

### 8.1 本机导出任务包

```powershell
python -m voicepack export-jobs "E:\mrxyy" --export-dir D:\taskpack_mrxyy
```

产物：

```
D:\taskpack_mrxyy\
├── manifest.json      # [{name,size,mtime,dur}...]，worker 靠它回填 size/mtime
├── HOWTO.txt
└── ogg\               # 4129 个 .ogg 原样复制（约 212MB）
```

打包上传（压缩包里 ogg 基本压不动，但**文件数从 4129 变成 1**，网盘/断点续传友好）：

```powershell
Compress-Archive -Path D:\taskpack_mrxyy\* -DestinationPath D:\mrxyy_taskpack.zip
```

### 8.2 远端跑

把 `mrxyy_taskpack.zip` 上传网盘/对象存储，在远端下载解压（或直接 `scp`）：

```bash
unzip -q mrxyy_taskpack.zip -d /root/taskpack
cd /root/vp
source .venv-remote/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python -m voicepack worker --jobs-dir /root/taskpack --out-jsonl /root/results.jsonl \
    --model large-v3 --device cuda
```

离线模式特点：

- 结果只写 `--out-jsonl` 指定的 JSONL，**已经写过的 name 会自动跳过**（中断后重跑接着来）；
- 不需要本机在线，跑完把 `results.jsonl`（约 3~5MB）下载下来即可；
- 任务包里 `manifest.json` 记录了每个 ogg 的 `size/mtime`，回传结果会带上，
  导入后 `AsrCache` 能直接命中，不会因为 mtime 对不上重跑。

### 8.3 本机导入

```powershell
python -m voicepack import-results "E:\mrxyy" .\results.jsonl
python -m voicepack filter "E:\mrxyy" --model large-v3
```

---

## 9. 命令速查

| 目的 | 命令 |
| --- | --- |
| 本机起服务 | `python -m voicepack serve <folder> --model large-v3 --port 8765 --host 127.0.0.1 --token <t>` |
| 本机建反向隧道 | `ssh -p <port> -R 8765:127.0.0.1:8765 -N root@<host>` |
| 远端 worker（在线） | `python -m voicepack worker --server http://127.0.0.1:8765 --token <t> --model large-v3 --device cuda` |
| 远端 worker（离线） | `python -m voicepack worker --jobs-dir <dir> --out-jsonl results.jsonl --model large-v3 --device cuda` |
| 下载全部结果 | `python -m voicepack pull --server http://127.0.0.1:8765 --token <t> --out-jsonl results.jsonl` |
| 合并进本地缓存 | `python -m voicepack import-results <folder> results.jsonl` |
| 导出离线任务包 | `python -m voicepack export-jobs <folder> --export-dir <dir>` |
| 看服务端进度 | 浏览器/curl 打开 `/api/status`（带 `?token=`） |
| 装远端环境 | `bash deploy/remote_setup.sh /root/vp` |

> 上表的 `python -m voicepack <子命令>` 是主用法；`python -m voicepack.remote <serve|worker|pull|export|import>`
> 是等价的模块入口（参数名略有差异：`--out` / `--cache-dir`，且支持 `--compute-type` 显式量化）。

worker 常用参数：

| 参数 | 说明 |
| --- | --- |
| `--workers N` | **并行下载线程数**（转写仍是单模型串行），GPU 建议 1，CPU 机型可 4~8 |
| `--chunk-sec S` | 每块目标音频秒数，显存小就调小（如 120） |
| `--max-clips N` | 每批任务数 / 每块句数上限（默认 32） |
| `--tmp-dir DIR` | 下载临时目录（默认自动建、结束删除） |
| `--out-jsonl FILE` | 结果 JSONL（在线模式是本地备份，离线模式是唯一产物） |

环境变量：`VOICEPACK_SERVER`、`VOICEPACK_TOKEN` 可以省掉 `--server/--token`；
`HF_ENDPOINT=https://hf-mirror.com` 用于模型下载加速。

---

## 10. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `鉴权失败 HTTP 401` | token 不一致：`serve` 和 `worker/pull` 的 `--token` 必须相同；`serve` 不传 token 会随机生成并打印，别用旧 token |
| 远端连不上 `127.0.0.1:8765` | 隧道没建/方向错了：必须是 `ssh -R`，且在本机执行；用 `-v` 看 `remote forward success` 日志 |
| `remote port forwarding failed` | 远端 8765 被占，换 `-R 18765:127.0.0.1:8765` 并让 worker 用 18765 |
| worker 卡在「等待服务端就绪」 | 本机 serve 没起来/防火墙拦截；`serve` 只监听 127.0.0.1 时必须走隧道 |
| `Could not locate cudnn_ops.so` / `libcublas.so.12` | CUDA 运行库缺失：重跑 `bash deploy/remote_setup.sh`（会装 nvidia-cublas-cu12 + nvidia-cudnn-cu12==9.*） |
| `CUDA out of memory` | 调小 `--chunk-sec`（240→120）和 `--max-clips`（32→16）；确认没有别的进程占显存（`nvidia-smi`） |
| 模型下载很慢/卡住 | 设 `HF_ENDPOINT=https://hf-mirror.com`；AutoDL 可 `source /etc/network_turbo` |
| RTF 比预期慢很多 | 看 `nvidia-smi` 利用率；CPU 解码可能成为瓶颈（`--workers` 只并行下载，块内解码已有 8 线程）；确认没开其他任务 |
| `磁盘满` | 任务包 + 模型约 3.5GB；容器盘建议 20GB；`--tmp-dir` 指到空间大的盘 |
| worker 被杀/实例被回收 | 直接重开一个实例重跑 worker：服务端缓存 + 离线 out_jsonl 都支持断点续跑，最多丢正在处理的那一批 |
| 服务端一直 pending>0 但没任务派发 | 任务卡在别的 worker 的约 4 分钟在途租约里，等一会儿或重跑 worker 即可 |
| 结果导入后本地仍要重转写 | `name/size/mtime/model` 有任一不匹配（比如文件被改过、模型名不一致）；这属于预期保护，不是 bug |
| 中文/emoji 打印乱码 | Windows 控制台需要 `set PYTHONIOENCODING=utf-8`；或 `chcp 65001` |

---

## 11. 安全与已知限制

### 安全

- **token 就是全部权限**：能列任务、能读语音文件、能写结果。公网直连时务必用长随机 token，
  并且只临时开放端口；更推荐 SSH 隧道（加密、不暴露端口）。
- `serve` 默认 `--host 0.0.0.0`；为了不裸奔，**不传 `--token` 时会自动生成随机 token 并打印**
  （`voicepack serve` 和 `voicepack.remote serve` 都是），worker/pull 必须带同一个；
  只想本机用就加 `--host 127.0.0.1`。
- 租来的机器上会短暂存放 ogg 语音文件（离线任务包模式会整个上传），跑完请清理实例；
  不要把这些文件放到公开可访问的存储。
- token 不要提交进 git、不要贴在截图里。

### 已知限制

- 服务端以**文件名（name）**判断「已完成」，同一文件夹内不能有重名文件；更换文件夹或改名后
  要换新的 `cache_path`（默认 `<folder>/voicepack_out/remote_inbox.jsonl`；
  `python -m voicepack serve` 则直接用本地 ASR 缓存名 `asr_<model>.jsonl`）。
- worker 转写失败的句子（ogg 损坏/解码失败）仍会回传一条空文本记录（附带 `error` 字段），
  避免无限重试；本地过滤会把空文本按「疑似非语言发声」处理，需要人工确认。
- 远端 worker 只做转写，不做过滤；词表和敏感判定不会离开本机。
- 在线模式每个 ogg 单独下载，不支持大文件分块续传（单个 ogg 一般 < 1MB，够用；
  HTTP 端点实现了 `Range`，但 worker 没做断点续传）。
- `large-v3` 模型必须与本地后续命令的 `--model` 一致，否则读不到缓存会重转写。
- 平台价格/可用性随时变化，本文估算仅供决策参考；以账单为准。
