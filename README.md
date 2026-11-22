# Subloom

先做好 CLI 的视频翻译工具。下载 YouTube、X 等 yt-dlp 支持的网站视频，转录、翻译，并导出带字幕的视频。支持本地视频。

首版保留原声，支持原文、译文、双语字幕；默认导出带可开关字幕轨道的 MP4，也可烧录硬字幕。MOSS 转录可保留说话人编号。目前不含配音、字幕编辑器或 GUI。

## 安装

需要 Python 3.11+、uv、yt-dlp、FFmpeg 和 ffprobe。本项目默认使用 Python 3.12。

```sh
uv sync --extra dev --extra native-build
uv run --extra native-build python scripts/setup_native.py
uv run subloom --help
uv run subloom doctor
```

macOS 上可以通过 Homebrew 安装 yt-dlp 和 FFmpeg。硬字幕需要带 libass 的构建；如果现有 FFmpeg 没有字幕滤镜，可安装 ffmpeg-full，并在配置中指定可执行文件的完整路径。软字幕不需要 libass。

## 配置模型

```sh
cp config.example.toml config.local.toml
export OPENAI_API_KEY='your-key'
uv run subloom doctor --config config.local.toml
```

转录和翻译分别配置，支持不同的 OpenAI 兼容服务、模型和 API key 环境变量。服务地址需要包含 API 前缀，如 https://api.openai.com/v1。密钥只从环境变量读取，不写入任务文件。

配置示例默认使用本地 whisper.cpp，直接读取 Vimpo 下载的 GGML 模型。转录不需要 API key，翻译默认使用 gpt-4o-mini，仍需配置模型服务。未指定配置或 --asr 时，保留原先的 OpenAI 转录默认值 whisper-1。

### 复用 Vimpo 的两个本地模型

setup_native.py 会在 .tools/ 下下载并编译固定版本的公开源码：whisper.cpp v1.9.5、transcribe.cpp v0.4.2。macOS 构建启用 Metal；原生库与权重独立于 Vimpo 应用，不读取或修改其应用框架。脚本不下载模型权重。首次构建需联网，完成后本地转录可离线运行。

默认读取以下两个文件，也可在配置中更改路径：

- ~/Library/Application Support/Vimpo/models/ggml-large-v3-turbo-q5_0.bin
- ~/Library/Application Support/Vimpo/models/MOSS-Transcribe-Diarize-Q5_K_M.gguf

```sh
uv run subloom transcribe ./input.mp4 --output ./transcript.json --asr whisper-cpp

uv run subloom transcribe ./input.mp4 --output ./moss.json --asr moss

uv run subloom run ./input.mp4 --asr whisper-cpp --style original

uv run subloom run ./input.mp4 --config config.local.toml

uv run subloom doctor --asr moss --json
```

whisper.cpp 的完整 JSON 包含 token 时间信息，适配器将英文 BPE 子词合并为单词，并保留可用的中文时间单元，再交给分句阶段。MOSS 当前只提供片段级时间戳，输出中的 speaker 字段保留原始说话人编号；当前没有词级对齐，分句阶段会提示并保留其原始片段。模型时间戳本身的精度仍取决于推理引擎。

单独 transcribe 会在输出 JSON 同名的 .work/ 目录保留标准化音频、引擎原始结果及日志。run 的原生日志和原始结果保存在任务目录。中断和超时会终止本次推理进程。更改模型、引擎文件、计算后端或线程配置会让转录缓存失效。native_backend 可设为 auto、metal 或 cpu；Whisper 的 metal 使用其 GPU 默认路径，cpu 显式禁用 GPU，MOSS 会将选项传给其后端选择器。

doctor 的 ASR ready 表示模型文件与可执行文件已找到，不能代替实际推理验证。完整 ready 还要求翻译服务的密钥已配置。

### 其他转录后端

--asr openai 使用 OpenAI 兼容转录服务。服务必须支持 /audio/transcriptions、verbose_json 和分段时间戳；准确分句还需要词级时间戳。CLI 会同时请求 segment 和 word 两种时间戳，并非所有转录模型都支持。音频转换为单声道 16kHz WAV，以不超过 600 秒的块上传，单块约 19.2MB。固定切块边界可能影响跨边界词句。

--asr local 使用可选的 faster-whisper：uv sync --extra local。默认 CPU/int8，首次使用可能下载 CTranslate2 格式的模型，不能直接加载上述 GGML/GGUF 文件。Apple Silicon 的 Metal 路径由 whisper-cpp 和 moss 后端提供。

## 使用

```sh
uv run subloom run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config.local.toml
uv run subloom run 'https://x.com/USER/status/POST_ID' --config config.local.toml

uv run subloom run ./input.mp4 --language en --target zh-CN --config config.local.toml

uv run subloom run 'URL' --cookies-from-browser chrome --config config.local.toml

uv run subloom run ./input.mp4 --mode hard --style translated --config config.local.toml

uv run subloom run ./input.mp4 --style original --config config.local.toml

uv run subloom run ./input.mp4 --transcript ./transcript.json --config config.local.toml

uv run subloom run ./input.mp4 --json-events --config config.local.toml
```

默认任务目录为 jobs/<基于来源和目标语言生成的ID>/，可用 --workspace 更改。重复执行同一命令会复用已完成且输入配置未变化的阶段；--force 重新执行阶段。单个任务加文件锁，避免多个进程同时写入。中断时释放文件锁；如果进程被强制杀死，可确认进程已退出后删除 .lock。

任务保存 source 视频、audio.wav、原始 transcript.json、分句后的 segments.json、translated.json、original.srt、translated.srt、captions.srt、output.mp4 和 job.json。修改 translated.json 后重新执行同一命令，会重新生成字幕和视频。重跑未完成的转录或翻译阶段可能重新请求该阶段的模型并产生费用。

软字幕可以在播放器中开关，部分网页播放器不会显示 MP4 内嵌字幕。需要字幕始终可见或发布到社交网站时，请导出硬字幕。硬字幕会重新编码视频。

## 字幕分句

默认流程是：转录 → 分句 → 翻译 → 生成字幕 → 导出。分句是独立的缓存阶段，调整分句配置不会重新转录。

分句使用真实词级时间戳，优先在句末标点和明显停顿处分开；超过长度或时长限制时，优先选择分句标点，再选择词边界。默认最长 6 秒、84 个显示列（中文等宽字符计 2 列），停顿阈值 0.6 秒。遇到 Dr.、Mr. 等常见英文缩写会避免直接断句。具体设置见 config.example.toml。

这是基于词级时间、标点和停顿的规则分句，尚未接入 Apple NaturalLanguage。长度和时长上限受词边界约束：单个词很长或多个词时间重叠时，会保留完整的词。模型未提供词级时间戳、或词无法与原文对应时，会输出警告并保留原分段，不按字符比例猜测语音时间。

可以单独重新分句，无需模型调用：

```sh
uv run subloom segment ./transcript.json \
  --output ./segments.json \
  --config config.local.toml
```

词级数据附在每个片段的 words 中，格式为：

```json
"words": [
  {"start": 0.0, "end": 0.5, "text": "Hello"},
  {"start": 0.5, "end": 1.0, "text": "world"}
]
```

旧任务需要重新转录才能获得词级数据；已有转录 JSON 也可直接导入。设置 segmenter = "none" 可以保留原始分段。已翻译的字幕请使用 render 导出，避免重新分句破坏人工修改的译文对应关系。

## 修改译文后直接导出

transcript.json 和 translated.json 使用相同的片段格式；后者附带 target_language：

```json
{
  "target_language": "zh-CN",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello, world.", "translation": "你好，世界。"}
  ]
}
```

```sh
uv run subloom render \
  --video ./input.mp4 \
  --segments ./translated.json \
  --output ./edited.mp4 \
  --mode soft
```

独立的 transcribe、segment、render 命令默认拒绝覆盖已有输出；确认替换时加 --overwrite。transcribe 和 segment 的输出必须使用 .json 扩展名，render 使用 .mp4。任务目录内由 run 管理的产物继续按阶段缓存规则更新。

render 不调用模型服务，可独立用于导出。字幕按稳定 ID 翻译，校验漏句、重复 ID、空译文；请求包含前文以帮助术语一致。接口暂时失败时指数退避重试，翻译格式不合法时尝试纠正一次。

## 后续 GUI

Python 核心在 src/subloom/pipeline.py，不依赖命令行解析。可先由桌面壳启动 CLI，读取 --json-events，将任务文件作为共享的数据格式，之后按需要增加本地 API。

建议下一阶段按实际使用顺序完成：任务列表与导入、阶段进度、视频预览与双语字幕编辑、模型设置、导出设置。批次缓存、术语表、按字幕剪辑、配音可逐项增加。

## 验证

```sh
make lint
make test
make check
SUBLOOM_TEST_NATIVE=1 uv run pytest tests/test_native.py -q
```

make lint 执行 Ruff 规则检查、格式检查和禁止注释检查。Python 代码禁止行注释、行尾注释、docstring 和独立字符串注释；TOML、YAML、Shell 和 Makefile 也禁止注释。字符串中的 # 按对应语言语法识别，不作为注释。检查覆盖项目源码、脚本、测试和配置，跳过 .tools、.venv、jobs、dist 等外部或生成目录。GitHub Actions 会在 push 和 pull request 时运行相同检查。

最后一条测试命令需要已编译的原生引擎和现有模型。

测试覆盖原生引擎 JSON/JSONL 解析、BPE 合词、说话人信息、模型缓存失效、超时终止、词级时间戳、自然分句、翻译校验、阶段恢复，以及本地测试服务和真实 FFmpeg 的完整 CLI 流程。默认跳过真实模型测试，显式启用后复用已安装的模型和引擎。测试不访问付费模型服务。
