# 最小验证 PoC（后端技术栈可行性）

验证目标：在 **Java 后端**跑通（与需求文档第 8 节选型一致，v1.3）

1. **离线中文 ASR**：Vosk 0.3.45（Maven Central，JNA 绑定）+ vosk-model-cn-0.22（大模型，选定）/ vosk-model-small-cn-0.22（小模型，对照）
2. **句向量匹配**：DJL 0.38.0 + ONNX Runtime + bge-small-zh-v1.5（CLS 向量 + 余弦相似度）

完整结论见 `docs/poc-report.md`。最初基于 sherpa-onnx + SenseVoice 验证（`PocMain`，保留作对照与备选方案），v1.3 起 ASR 选型切换为 Vosk（`VoskMain`）。

## 目录结构

```
poc/
├── pom.xml                       # Maven 工程（Java 17）；exec.main.class 可切换主类
├── setup_jars.sh                 # [sherpa 备选方案] 下载 sherpa-onnx jar 并装本地仓库
├── download_models.sh            # [sherpa 备选方案] 下载 SenseVoice / bge 模型
├── tts_gen.ps1 + tts_phrases.txt # Windows SAPI(Huihui) 生成 10 条 16k 测试音频
├── tools/OrtLoadTest.java        # ORT/sherpa native 共存性诊断工具
├── libs/vosk/vosk-win64-0.3.45/  # Vosk 官方 win64 native（libvosk.dll + MinGW 运行库）
├── libs/  models/  audio/        # 产物，不入库（见 .gitignore）
└── src/main/java/poc/
    ├── VoskMain.java             # ★ Vosk 方案主验证（参数：模型目录名，默认小模型）
    ├── PocMain.java              # sherpa+SenseVoice 方案（备选，对照）
    └── VoskMemTest.java          # Vosk 模型常驻内存采样
```

## 复现步骤

```bash
# 0) 环境：JDK 17+、Maven 3.8+（本机验证：17.0.14 / 3.8.4，Windows 11 x64）

# 1) Vosk native（win64 官方包，任一可达通道）
curl -L -o vosk-win64-0.3.45.zip \
  https://alphacephei.com/vosk/win64/vosk-win64-0.3.45.zip   # 或 GitHub Release
unzip vosk-win64-0.3.45.zip -d libs/vosk/

# 2) 模型
#    小模型（44MB，对照用）：alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip → 解压到 models/
#    大模型（1.3GB，选定）：
curl -L -o vosk-model-cn-0.22.zip \
  https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
#   （慢时可走镜像：https://hf-mirror.com/LiangJingyi/vosk-model-cn-0.22/resolve/main/model-cn.zip，
#     注意镜像包内根目录为 model-cn，解压到 models/ 即可）
unzip vosk-model-cn-0.22.zip -d models/
#    bge 句向量模型（90MB）：
mkdir -p models/bge-small-zh
curl -L -o models/bge-small-zh/model.onnx     https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx
curl -L -o models/bge-small-zh/tokenizer.json https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json

# 3) 测试音频（需 Windows + 中文 SAPI 语音 Microsoft Huihui Desktop）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File poc/tts_gen.ps1

# 4) 运行（★ 注意 jna.encoding=UTF-8，否则 Windows 下 Vosk 中文结果乱码）
cd poc
MAVEN_OPTS="-Dfile.encoding=UTF-8 -Djna.encoding=UTF-8" \
  mvn -q compile exec:java "-Dexec.main.class=poc.VoskMain" "-Dexec.args=model-cn"
#    小模型：把 exec.args 换成 vosk-model-small-cn-0.22（或省略，默认小模型）
#    sherpa 备选方案：mvn -q compile exec:java（默认主类 PocMain，需先 bash setup_jars.sh）
```

## 关键集成点（重要）

1. **Vosk 中文编码**：JNA 默认按平台编码（Windows=GBK）解码 native 字符串，必须设置 `jna.encoding=UTF-8`，否则识别结果是乱码（Spring Boot 中在启动最早处 `System.setProperty` 或 JVM 参数均可）。
2. **Vosk 无 ITN**：数字输出保持中文（"二十六"），需求 FR-3 第 4 步"中文数字→阿拉伯数字"为**必做**（非兜底）。
3. **ORT（句向量侧）Windows 集成**：`com.microsoft.onnxruntime` 官方包的 win-x64 核心 DLL 在本机初始化失败（Win32 1114，与 ASR 引擎无关）；解法是把可用的 `onnxruntime.dll` + 官方 `onnxruntime4j_jni.dll` 放同一目录并设 `onnxruntime.native.path`（`VoskMain#prepareOrtNativeDir` 已实现，当前借用 sherpa native 包内验证可用的 DLL；若彻底去 sherpa 化，需将这份 DLL 作为独立资产托管）。
4. **sherpa 备选方案依赖**：官方不发布 Maven Central，仅 GitHub Release jar，需 `setup_jars.sh` + 私服托管；Maven 3.8 需固定 compiler 插件 3.13+。

## 国产化迁移要点

- Vosk 官方提供 linux x86_64 / aarch64 / armv7l / riscv64 native：飞腾/鲲鹏/海光/兆芯（麒麟 V10、统信 UOS）**换 so + 改 `jna.library.path` 即可，Java 代码零改动**；
- 龙芯 LoongArch 无官方构建，需源码自编译（Kaldi 体系）；句向量侧 onnxruntime 同样无官方 loongarch 包（Loongnix 有社区构建，经 `onnxruntime.native.path` 挂载）；
- 两引擎均为 Apache-2.0 许可，无授权风险。详见 `docs/poc-report.md`。
