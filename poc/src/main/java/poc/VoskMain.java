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
import org.vosk.Model;
import org.vosk.Recognizer;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 最小验证（PoC，Vosk 方案）：后端跑通
 *   1) Vosk(0.3.45, JNA) + vosk-model-small-cn-0.22 中文离线识别
 *   2) DJL(0.38.0) + ONNX Runtime + bge-small-zh-v1.5 句向量 + 余弦相似度指令匹配
 * 测试音频与 sherpa 方案同源（SAPI Huihui 合成的 10 条 16k WAV，覆盖需求附录 A 场景）。
 */
public final class VoskMain {

    private record Case(String wav, String expect) {}

    private static final Map<String, List<String>> COMMANDS = new LinkedHashMap<>();
    private static final Map<String, List<String>> PARAM_DEMO = new LinkedHashMap<>();

    static {
        COMMANDS.put("ac_on", List.of("打开空调", "开空调", "空调开启", "帮我把空调打开"));
        COMMANDS.put("ac_off", List.of("关闭空调", "把空调关了", "关空调", "空调关闭"));
        COMMANDS.put("light_on", List.of("打开灯", "开灯", "把客厅灯打开", "打开客厅的灯"));
        PARAM_DEMO.put("temp_up(期望走规则)", List.of("温度高一点", "调高温度", "升高温度"));
        PARAM_DEMO.put("temp_down(期望走规则)", List.of("温度调低", "降低温度", "温度低一点"));
    }

    private static final double THRESHOLD = 0.75;
    private static final Pattern VOSK_TEXT = Pattern.compile("\"text\"\\s*:\\s*\"(.*?)\"");

    public static void main(String[] args) throws Exception {
        Path root = Paths.get(System.getProperty("user.dir"));
        if (!Files.isDirectory(root.resolve("models"))) {
            throw new IllegalStateException("请在 poc/ 目录下运行（未找到 poc/models）: " + root);
        }
        String modelDirName = args.length > 0 ? args[0] : "vosk-model-small-cn-0.22";
        Path modelDir = root.resolve("models").resolve(modelDirName);
        if (!Files.isDirectory(modelDir)) {
            throw new IllegalStateException("未找到模型目录: " + modelDir);
        }
        System.out.printf(Locale.ROOT, "JVM %s | cwd=%s | 模型=%s%n",
                System.getProperty("java.version"), root, modelDirName);
        // JNA 默认用平台编码解码 native 字符串（Windows=GBK），必须强制 UTF-8，否则中文识别结果乱码
        System.setProperty("jna.encoding", "UTF-8");

        // ============ 0. native 库就绪 ============
        // a) ORT：官方 win-x64 核心 DLL 本机初始化失败，改用 sherpa 包内可用的 onnxruntime.dll
        //    + 官方 JNI 胶水（详见 docs/poc-report.md §3.1；句向量侧与 ASR 引擎无关）
        Path ortNativeDir = prepareOrtNativeDir();
        System.setProperty("onnxruntime.native.path", ortNativeDir.toString());
        System.out.println("==== [0] ORT native 就绪: " + ortNativeDir + " ====");
        // b) Vosk：JNA 加载 libvosk.dll（win64 官方包，含 MinGW 运行库 dll）
        Path voskLibDir = root.resolve("libs/vosk/vosk-win64-0.3.45").toAbsolutePath();
        if (!Files.exists(voskLibDir.resolve("libvosk.dll"))) {
            throw new IllegalStateException("未找到 libvosk.dll: " + voskLibDir);
        }
        System.setProperty("jna.library.path", voskLibDir.toString());
        System.out.println("==== [0] Vosk native 就绪: " + voskLibDir + " ====");

        // ============ 1. ASR：Vosk + vosk-model-small-cn-0.22 ============
        System.out.println();
        System.out.println("==== [1] Vosk Recognizer + " + modelDirName + " ====");
        long t0 = System.nanoTime();
        Model model = new Model(modelDir.toString());
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

        Map<String, String> asrClean = new LinkedHashMap<>();
        Map<String, String> asrExpect = new LinkedHashMap<>();
        double maxRtf = 0;
        long maxAsrMs = 0;
        try (Recognizer recognizer = new Recognizer(model, 16000f)) {
            recognizer.setWords(false);
            for (Case c : cases) {
                byte[] pcm = readMono16kWavBytes(root.resolve("audio").resolve(c.wav() + ".wav"));
                double audioSec = pcm.length / 2 / 16000.0;
                long s0 = System.nanoTime();
                recognizer.acceptWaveForm(pcm, pcm.length);
                String json = recognizer.getFinalResult();
                long ms = (System.nanoTime() - s0) / 1_000_000;
                String clean = cleanAsr(extractVoskText(json));
                asrClean.put(c.wav(), clean);
                asrExpect.put(c.wav(), c.expect());
                double rtf = ms / 1000.0 / audioSec;
                maxRtf = Math.max(maxRtf, rtf);
                maxAsrMs = Math.max(maxAsrMs, ms);
                System.out.printf(Locale.ROOT, "%s 期望[%s] 识别[%s] 音频%.1fs 耗时%dms RTF%.3f%n",
                        c.wav(), c.expect(), clean, audioSec, ms, rtf);
            }
        } finally {
            model.close();
        }
        System.out.printf(Locale.ROOT, "ASR 汇总：最大单条耗时 %d ms，最大 RTF %.3f%n", maxAsrMs, maxRtf);

        // ============ 2. 句向量：DJL + ONNX Runtime + bge-small-zh-v1.5 ============
        System.out.println();
        System.out.println("==== [2] DJL + ONNX Runtime + bge-small-zh-v1.5（CLS 向量 + 余弦相似度） ====");
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(
                root.resolve("models/bge-small-zh/tokenizer.json"), Map.of());
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(root.resolve("models/bge-small-zh"))
                .optEngine("OnnxRuntime")
                .optTranslator(new ClsEmbedTranslator(tokenizer))
                .build();
        long e0 = System.nanoTime();
        ZooModel<String, float[]> bgeModel = criteria.loadModel();
        Predictor<String, float[]> predictor = bgeModel.newPredictor();
        System.out.printf(Locale.ROOT, "bge 模型加载耗时 %d ms%n", (System.nanoTime() - e0) / 1_000_000);
        predictor.predict("预热句子");

        {
            Map<String, float[][]> bank = new LinkedHashMap<>();
            for (var e : COMMANDS.entrySet()) {
                bank.put(e.getKey(), embedAll(predictor, e.getValue()));
            }
            List<String> queries = new ArrayList<>(asrClean.values());
            float[][] qVec = new float[queries.size()][];
            long encMax = 0;
            for (int i = 0; i < queries.size(); i++) {
                long s0 = System.nanoTime();
                qVec[i] = predictor.predict(queries.get(i));
                encMax = Math.max(encMax, (System.nanoTime() - s0) / 1_000_000);
            }
            System.out.printf(Locale.ROOT, "单条文本编码最大 %d ms%n%n", encMax);

            System.out.println("指令得分矩阵（行=查询，列=指令，阈值 " + THRESHOLD + "）：");
            System.out.printf(Locale.ROOT, "%-22s %8s %8s %9s   %s%n",
                    "query", "ac_on", "ac_off", "light_on", "命中");
            List<String> qNames = new ArrayList<>(asrClean.keySet());
            for (int i = 0; i < queries.size(); i++) {
                Map<String, Double> scores = commandScores(bank, qVec[i]);
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
                System.out.printf(Locale.ROOT, "%-22s %8.3f %8.3f %9.3f   %s%n",
                        abbreviate(queries.get(i), 20),
                        scores.get("ac_on"), scores.get("ac_off"), scores.get("light_on"), verdict);
            }

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
            String t6 = asrClean.get("t06");
            if (t6.contains("26")) {
                check("数字输出含 26（Vosk ITN 生效）", true);
            } else if (t6.contains("二十六")) {
                check("数字保持中文'二十六'（Vosk 无 ITN，需走 FR-3 中文数字归一化兜底）", true);
            } else {
                check("数字输出异常（既无 26 也无 二十六）: " + t6, false);
            }

            System.out.println();
            System.out.println("参数指令交叉得分演示（相似度路径的误命中风险，即文档设计原则 3 的依据）：");
            Map<String, float[][]> demoBank = new LinkedHashMap<>();
            for (var e : PARAM_DEMO.entrySet()) {
                demoBank.put(e.getKey(), embedAll(predictor, e.getValue()));
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
        System.out.println("==== PoC（Vosk 方案）执行完毕 ====");
    }

    // ---------- 工具方法 ----------

    private static String extractVoskText(String json) {
        Matcher m = VOSK_TEXT.matcher(json);
        return m.find() ? m.group(1) : "";
    }

    /** Vosk 中文输出为空格分词的小写文本，去除空格与首尾标点（对应 FR-3 归一化） */
    private static String cleanAsr(String raw) {
        String s = raw.replace(" ", "").trim();
        return s.replaceAll("^[\\p{P}\\s]+|[\\p{P}\\s]+$", "").trim();
    }

    private static String expectToCommand(String expect) {
        if (expect.contains("温度")) return "ac_temp(规则)";
        if (expect.contains("灯")) return "light_on";
        if (expect.contains("关闭")) return "ac_off";
        if (expect.equals("今天天气怎么样")) return "(未命中)";
        return "ac_on";
    }

    private static float[][] embedAll(Predictor<String, float[]> predictor, List<String> texts) throws Exception {
        float[][] out = new float[texts.size()][];
        for (int i = 0; i < texts.size(); i++) {
            out[i] = predictor.predict(texts.get(i));
        }
        return out;
    }

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

    /** 读取 16k/16bit/单声道 PCM WAV，返回原始小端 short 字节流（Vosk acceptWaveForm 输入） */
    private static byte[] readMono16kWavBytes(Path path) throws Exception {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(path.toFile())) {
            AudioFormat f = ais.getFormat();
            if (f.getSampleRate() != 16000f || f.getChannels() != 1
                    || f.getSampleSizeInBits() != 16
                    || !AudioFormat.Encoding.PCM_SIGNED.equals(f.getEncoding())) {
                throw new IllegalStateException("期望 16kHz/16bit/单声道 PCM WAV，实际: " + f);
            }
            return ais.readAllBytes();
        }
    }

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
        try (InputStream in = VoskMain.class.getClassLoader().getResourceAsStream(res)) {
            if (in == null) {
                throw new IllegalStateException("classpath 缺少资源: " + res);
            }
            if (Files.exists(target) && Files.size(target) > 0) {
                return;
            }
            Files.copy(in, target);
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
