package poc;

import org.vosk.Model;

/** 临时工具：加载 Vosk 模型后驻留，用于采样常驻内存 */
public class VoskMemTest {
    public static void main(String[] args) throws Exception {
        System.setProperty("jna.encoding", "UTF-8");
        long t0 = System.nanoTime();
        Model model = new Model(args[0]);
        System.out.println("MODEL_LOADED_MS=" + (System.nanoTime() - t0) / 1_000_000);
        Thread.sleep(90_000);
        model.close();
    }
}
