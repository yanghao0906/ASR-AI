import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.k2fsa.sherpa.onnx.LibraryUtils;

/** 诊断 sherpa-onnx 与 DJL/OR T 两个 native 库在同一进程内的共存性（单文件直接运行） */
public class OrtLoadTest {
    public static void main(String[] args) throws Exception {
        String mode = args[0]; // ort | sherpa | ort-then-sherpa | sherpa-then-ort
        String modelPath = args.length > 1 ? args[1] : null;
        if (mode.startsWith("sherpa")) {
            System.out.println("--> LibraryUtils.load() 加载 sherpa natives");
            LibraryUtils.load();
            System.out.println("<-- sherpa natives OK");
        }
        if (!mode.equals("sherpa")) {
            System.out.println("--> ORT createSession: " + modelPath);
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (OrtSession s = env.createSession(modelPath, new OrtSession.SessionOptions())) {
                System.out.println("<-- ORT session OK, inputs=" + s.getInputInfo().keySet());
            }
        }
        if (mode.equals("ort-then-sherpa")) {
            System.out.println("--> LibraryUtils.load() 后加载 sherpa natives");
            LibraryUtils.load();
            System.out.println("<-- sherpa natives OK");
        }
        System.out.println("MODE[" + mode + "] PASS");
    }
}
