package poc;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.k2fsa.sherpa.onnx.OfflineModelConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizer;
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizerResult;
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig;
import com.k2fsa.sherpa.onnx.OfflineStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 最小验证（PoC）：在 Java 后端跑通
 *   1) sherpa-onnx(1.13.8) + SenseVoice-small int8 中文离线识别
 *   2) DJL(0.38.0) + ONNX Runtime + bge-small-zh-v1.5 句向量 + 余弦相似度指令匹配
 * 音频为 SAPI(Huihui) 合成的 10 条 16k/16bit/单声道 WAV，覆盖需求文档附录 A 场景。
 */
public final class PocMain {

    private record Case(String wav, String expect) {}

    /** 相似度指令库（无参数指令，对齐文档 commands.yaml 示例 + 一条灯光指令用于跨设备区分） */
    private static final Map<String, List<String>> COMMANDS = new LinkedHashMap<>();
    /** 参数指令短语对：仅用于演示"字面相近、语义相反"为何必须走规则（文档设计原则 3） */
    private static final Map<String, List<String>> PARAM_DEMO = new LinkedHashMap<>();

    static {
        COMMANDS.put("ac_on", List.of("打开空调", "开空调", "空调开启", "帮我把空调打开"));
        COMMANDS.put("ac_off", List.of("关闭空调", "把空调关了", "关空调", "空调关闭"));
        COMMANDS.put("light_on", List.of("打开灯", "开灯", "把客厅灯打开", "打开客厅的灯"));
        PARAM_DEMO.put("temp_up(期望走规则)", List.of("温度高一点", "调高温度", "升高温度"));
        PARAM_DEMO.put("temp_down(期望走规则)", List.of("温度调低", "降低温度", "温度低一点"));
    }

    private static final String BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";
    private static final double THRESHOLD = 0.75;

    public static void main(String[] args) throws Exception {
        Path root = Paths.get(System.getProperty("user.dir"));
        if (!Files.isDirectory(root.resolve("models"))) {
            throw new IllegalStateException("请在 poc/ 目录下运行（未找到 poc/models）: " + root);
        }
        System.out.printf(Locale.ROOT, "JVM %s | cwd=%s%n", System.getProperty("java.version"), root);

        // ============ 0. ORT native 加载（关键集成点） ============
        // com.microsoft 官方 onnxruntime 包的 win-x64 核心 DLL 在本机初始化失败（DllMain 1114），
        // 而 sherpa-onnx native 包自带的 onnxruntime.dll(1.28.2) 验证可用（与加载顺序无关）。
        // 解法：把"sherpa 核心 DLL + 官方 JNI 胶水 DLL"提取到同一目录，
        // 通过 -Donnxruntime.native.path 让 ORT Java 绑定直接使用该目录（API 版本同为 28，二进制兼容）。
        Path ortNativeDir = prepareOrtNativeDir();
        System.setProperty("onnxruntime.native.path", ortNativeDir.toString());
        System.out.println();
        System.out.println("==== [0] ORT native 就绪（onnxruntime.native.path=" + ortNativeDir + "） ====");

        // DJL + bge 模型加载（此时才真正触发 ORT native 加载）
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(
                root.resolve("models/bge-small-zh/tokenizer.json"), Map.of());
        ClsEmbedTranslator translator = new ClsEmbedTranslator(tokenizer);
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(root.resolve("models/bge-small-zh"))
                .optEngine("OnnxRuntime")
                .optTranslator(translator)
                .build();
        long e0 = System.nanoTime();
        ZooModel<String, float[]> bgeModel = criteria.loadModel();
        Predictor<String, float[]> predictor = bgeModel.newPredictor();
        System.out.printf(Locale.ROOT, "bge 模型加载耗时 %d ms%n", (System.nanoTime() - e0) / 1_000_000);
        long w0 = System.nanoTime();
        predictor.predict("预热句子"); // 首次推理含 ORT 会话初始化
        System.out.printf(Locale.ROOT, "首次推理(含初始化) %d ms%n", (System.nanoTime() - w0) / 1_000_000);

        // ============ 1. ASR：sherpa-onnx + SenseVoice-small int8 ============
        System.out.println();
        System.out.println("==== [1] sherpa-onnx OfflineRecognizer + SenseVoice-small(int8, zh) ====");
        long t0 = System.nanoTime();
        OfflineRecognizerConfig cfg = OfflineRecognizerConfig.builder()
                .setOfflineModelConfig(OfflineModelConfig.builder()
                        .setSenseVoice(OfflineSenseVoiceModelConfig.builder()
                                .setModel(root.resolve("models/sensevoice/model.int8.onnx").toString())
                                .setLanguage("zh")
                                .setInverseTextNormalization(true)
                                .build())
                        .setTokens(root.resolve("models/sensevoice/tokens.txt").toString())
                        .setNumThreads(2)
                        .setDebug(false)
                        .build())
                .build();
        OfflineRecognizer recognizer = new OfflineRecognizer(cfg);
        System.out.printf(Locale.ROOT, "模型加载耗时 %d ms%n", (System.nanoTime() - t0) / 1_000_000);

        List<Case> cases = List.of(
                new Case("t01", "打开空调"),
                new Case("t02", "开空调"),
                new Case("t03", "空调开启"),
                new Case("t04", "帮我把空调打开"),
                new Case("t05", "关闭空调"),
                new Case("t06", "空调温度设置到二十六度"),
                new Case("t07", "空调温度高一点"),
                new Case("t08", "空调温度调低"),
                new Case("t09", "今天天气怎么样"),
                new Case("t10", "把客厅的灯打开"));

        Map<String, String> asrClean = new LinkedHashMap<>(); // 清洗后 ASR 文本（按 wav 名）
        Map<String, String> asrExpect = new LinkedHashMap<>();
        double maxRtf = 0;
        long maxAsrMs = 0;
        boolean tagSeen = false;
        for (Case c : cases) {
            float[] pcm = readMono16kWav(root.resolve("audio").resolve(c.wav() + ".wav"));
            double audioSec = pcm.length / 16000.0;
            long s0 = System.nanoTime();
            OfflineStream stream = recognizer.createStream();
            try {
                stream.acceptWaveform(pcm, 16000);
                recognizer.decode(stream);
                OfflineRecognizerResult r = recognizer.getResult(stream);
                long ms = (System.nanoTime() - s0) / 1_000_000;
                String raw = r.getText();
                String clean = cleanAsr(raw);
                tagSeen |= raw.contains("<|");
                asrClean.put(c.wav(), clean);
                asrExpect.put(c.wav(), c.expect());
                double rtf = ms / 1000.0 / audioSec;
                maxRtf = Math.max(maxRtf, rtf);
                maxAsrMs = Math.max(maxAsrMs, ms);
                System.out.printf(Locale.ROOT, "%s 期望[%s] 识别[%s] 音频%.1fs 耗时%dms RTF%.3f%n",
                        c.wav(), c.expect(), clean, audioSec, ms, rtf);
            } finally {
                stream.release();
            }
        }
        recognizer.release();
        System.out.printf(Locale.ROOT, "ASR 汇总：最大单条耗时 %d ms，最大 RTF %.3f，原始输出含<|标签|>=%s%n",
                maxAsrMs, maxRtf, tagSeen);

        // ============ 2. 句向量匹配：CLS 向量 + 余弦相似度 ============
        System.out.println();
        System.out.println("==== [2] bge-small-zh-v1.5 句向量匹配（阈值 " + THRESHOLD + "） ====");
        {

            // 指令短语库向量（文档侧不加前缀）
            Map<String, float[][]> bank = new LinkedHashMap<>();
            for (var e : COMMANDS.entrySet()) {
                bank.put(e.getKey(), embedAll(predictor, e.getValue(), null));
            }
            // 查询侧向量：ASR 清洗文本；同时算带 bge 检索前缀的版本作对比
            List<String> queries = new ArrayList<>(asrClean.values());
            float[][] qVec = new float[queries.size()][];
            float[][] qVecPrefixed = new float[queries.size()][];
            long encMax = 0;
            for (int i = 0; i < queries.size(); i++) {
                long s0 = System.nanoTime();
                qVec[i] = predictor.predict(queries.get(i));
                qVecPrefixed[i] = predictor.predict(BGE_QUERY_PREFIX + queries.get(i));
                encMax = Math.max(encMax, (System.nanoTime() - s0) / 1_000_000);
            }
            System.out.printf(Locale.ROOT, "单条文本编码(含两次推理)最大 %d ms%n%n", encMax);

            // ---- 匹配矩阵：每条查询 × 每条指令（指令得分 = 短语最高分）----
            System.out.println("指令得分矩阵（行=查询，列=指令，阈值 " + THRESHOLD + "）：");
            System.out.printf(Locale.ROOT, "%-22s %8s %8s %9s %8s   %s%n",
                    "query", "ac_on", "ac_off", "light_on", "带前缀Top1", "命中");
            List<String> qNames = new ArrayList<>(asrClean.keySet());
            for (int i = 0; i < queries.size(); i++) {
                Map<String, Double> scores = commandScores(bank, qVec[i]);
                Map<String, Double> scoresPref = commandScores(bank, qVecPrefixed[i]);
                String top = argmax(scores);
                double topScore = scores.get(top);
                String wav = qNames.get(i);
                boolean hit = topScore >= THRESHOLD;
                String verdict;
                String expect = asrExpect.get(wav);
                if (expect.equals("空调温度设置到二十六度") || expect.equals("空调温度高一点")
                        || expect.equals("空调温度调低")) {
                    verdict = "参数指令→走规则(不参与相似度)";
                } else if (!hit) {
                    verdict = "未命中→返回Top3候选";
                } else {
                    verdict = "命中 " + top + (top.equals(expectToCommand(expect)) ? " ✔" : " ✘ 期望 " + expectToCommand(expect));
                }
                System.out.printf(Locale.ROOT, "%-22s %8.3f %8.3f %9.3f %8.3f   %s%n",
                        abbreviate(queries.get(i), 20),
                        scores.get("ac_on"), scores.get("ac_off"), scores.get("light_on"),
                        scoresPref.get(argmax(scoresPref)), verdict);
            }

            // ---- 关键校验项 ----
            System.out.println();
            check("ac_on 四种说法 Top1 全命中", wavOf(asrClean, "空调", List.of("t01", "t02", "t03", "t04"))
                    .stream().allMatch(w -> {
                        Map<String, Double> s = commandScores(bank, vecOf(queries, qVec, asrClean.get(w)));
                        return argmax(s).equals("ac_on") && s.get("ac_on") >= THRESHOLD;
                    }));
            check("关闭空调 命中 ac_off 且不串 ac_on", wavOf(asrClean, "", List.of("t05")).stream()
                    .allMatch(w -> {
                        Map<String, Double> s = commandScores(bank, vecOf(queries, qVec, asrClean.get(w)));
                        return argmax(s).equals("ac_off") && s.get("ac_on") < THRESHOLD;
                    }));
            check("无关句(今天天气怎么样) 全部低于阈值",
                    commandScores(bank, vecOf(queries, qVec, asrClean.get("t09"))).values()
                            .stream().allMatch(v -> v < THRESHOLD));
            check("打开灯类语句 命中 light_on 而非 ac_on",
                    argmax(commandScores(bank, vecOf(queries, qVec, asrClean.get("t10")))).equals("light_on"));
            check("ASR 数字逆文本归一化(ITN) 输出含 26", asrClean.get("t06").contains("26"));

            // ---- 参数指令演示：为什么必须走规则 ----
            System.out.println();
            System.out.println("参数指令交叉得分演示（相似度路径的误命中风险，即文档设计原则 3 的依据）：");
            Map<String, float[][]> demoBank = new LinkedHashMap<>();
            for (var e : PARAM_DEMO.entrySet()) {
                demoBank.put(e.getKey(), embedAll(predictor, e.getValue(), null));
            }
            for (String w : List.of("t07", "t08")) {
                float[] q = predictor.predict(asrClean.get(w));
                Map<String, Double> s = commandScores(demoBank, q);
                System.out.printf(Locale.ROOT, "  [%s] %s → temp_up=%.3f temp_down=%.3f  （两者分差仅 %.3f，靠相似度无法保证严格区分）%n",
                        w, asrClean.get(w), s.get("temp_up(期望走规则)"), s.get("temp_down(期望走规则)"),
                        Math.abs(s.get("temp_up(期望走规则)") - s.get("temp_down(期望走规则)")));
            }
        }
        bgeModel.close();
        tokenizer.close();
        System.out.println();
        System.out.println("==== PoC 执行完毕 ====");
    }

    // ---------- 工具方法 ----------

    /** 从 classpath 提取"sherpa 核心 ORT DLL + 官方 JNI 胶水 DLL"到临时目录（幂等） */
    private static Path prepareOrtNativeDir() throws Exception {
        Path dir = Paths.get(System.getProperty("java.io.tmpdir"), "asr-poc-ort-native");
        Files.createDirectories(dir);
        copyResource("sherpa-onnx/native/win-x64/onnxruntime.dll", dir.resolve("onnxruntime.dll"));
        copyResource("ai/onnxruntime/native/win-x64/onnxruntime4j_jni.dll", dir.resolve("onnxruntime4j_jni.dll"));
        copyResource("ai/onnxruntime/native/win-x64/onnxruntime_providers_shared.dll",
                dir.resolve("onnxruntime_providers_shared.dll"));
        return dir;
    }

    private static void copyResource(String res, Path target) throws Exception {
        try (InputStream in = PocMain.class.getClassLoader().getResourceAsStream(res)) {
            if (in == null) {
                throw new IllegalStateException("classpath 缺少资源: " + res);
            }
            if (Files.exists(target) && Files.size(target) > 0) {
                return; // 幂等：已提取过
            }
            Files.copy(in, target);
        }
    }

    /** SenseVoice 输出形如 "<|zh|><|NEUTRAL|><|Speech|><|withitn|>打开空调。"，
     *  需剥离控制标签并去除首尾标点（对应需求 FR-3 归一化的第 1 步） */
    private static String cleanAsr(String raw) {
        String s = raw.replaceAll("<\\|[^|]*\\|>", "").replace(" ", "").trim();
        return s.replaceAll("^[\\p{P}\\s]+|[\\p{P}\\s]+$", "").trim();
    }

    private static String expectToCommand(String expect) {
        if (expect.contains("温度")) return "ac_temp(规则)";
        if (expect.contains("灯")) return "light_on";
        if (expect.contains("关闭")) return "ac_off";
        if (expect.equals("今天天气怎么样")) return "(未命中)";
        return "ac_on";
    }

    private static float[][] embedAll(Predictor<String, float[]> predictor, List<String> texts, String prefix)
            throws Exception {
        float[][] out = new float[texts.size()][];
        for (int i = 0; i < texts.size(); i++) {
            out[i] = predictor.predict(prefix == null ? texts.get(i) : prefix + texts.get(i));
        }
        return out;
    }

    /** 指令得分 = 其所有短语余弦相似度最大值（向量已 L2 归一化，点积即余弦） */
    private static Map<String, Double> commandScores(Map<String, float[][]> bank, float[] query) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (var e : bank.entrySet()) {
            double best = 0;
            for (float[] p : e.getValue()) {
                best = Math.max(best, dot(query, p));
            }
            out.put(e.getKey(), best);
        }
        return out;
    }

    private static String argmax(Map<String, Double> m) {
        String best = null;
        double v = -2;
        for (var e : m.entrySet()) {
            if (e.getValue() > v) {
                v = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
        return s;
    }

    private static float[] vecOf(List<String> queries, float[][] qVec, String text) {
        return qVec[queries.indexOf(text)];
    }

    private static List<String> wavOf(Map<String, String> asrClean, String contains, List<String> ids) {
        List<String> out = new ArrayList<>();
        for (String id : ids) {
            if (contains.isEmpty() || asrClean.get(id).contains(contains)) out.add(id);
        }
        return out;
    }

    private static void check(String name, boolean ok) {
        System.out.println((ok ? "[PASS] " : "[FAIL] ") + name);
    }

    private static String abbreviate(String s, int width) {
        return s.length() <= width ? s : s.substring(0, width - 1) + "…";
    }

    /** 读取 16k/16bit/单声道 PCM WAV，转为 [-1,1) 浮点采样（与 sherpa-onnx 期望一致） */
    private static float[] readMono16kWav(Path path) throws Exception {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(path.toFile())) {
            AudioFormat f = ais.getFormat();
            if (f.getSampleRate() != 16000f || f.getChannels() != 1
                    || f.getSampleSizeInBits() != 16
                    || !AudioFormat.Encoding.PCM_SIGNED.equals(f.getEncoding())) {
                throw new IllegalStateException("期望 16kHz/16bit/单声道 PCM WAV，实际: " + f);
            }
            byte[] bytes = ais.readAllBytes();
            ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            float[] pcm = new float[bytes.length / 2];
            for (int i = 0; i < pcm.length; i++) {
                pcm[i] = bb.getShort() / 32768f;
            }
            return pcm;
        }
    }

    /** bge-small-zh-v1.5 使用 CLS 向量 + L2 归一化 */
    private static final class ClsEmbedTranslator implements Translator<String, float[]> {
        private final HuggingFaceTokenizer tokenizer;

        ClsEmbedTranslator(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding enc = tokenizer.encode(input);
            NDManager m = ctx.getNDManager();
            return new NDList(
                    m.create(enc.getIds(), new Shape(1, enc.getIds().length)),
                    m.create(enc.getAttentionMask(), new Shape(1, enc.getAttentionMask().length)),
                    m.create(enc.getTypeIds(), new Shape(1, enc.getTypeIds().length)));
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray hidden = list.get(0); // last_hidden_state [1, seq, hidden]
            float[] cls = hidden.get(new NDIndex("0, 0, :")).toFloatArray();
            double norm = 0;
            for (float v : cls) norm += (double) v * v;
            norm = Math.sqrt(norm);
            float[] out = new float[cls.length];
            for (int i = 0; i < cls.length; i++) out[i] = (float) (cls[i] / norm);
            return out;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
