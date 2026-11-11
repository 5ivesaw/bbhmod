package dev.bbh.agent.runtime;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import dev.bbh.agent.core.Policy;
import dev.bbh.agent.core.PolicyOutput;
import dev.bbh.agent.core.Spec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Map;

/** Runs policy.onnx on the CPU with one thread, carrying the GRU hidden state between steps. */
public final class OnnxPolicy implements Policy {
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final Spec spec;
    private float[] hidden;

    public OnnxPolicy(byte[] model, Spec spec) throws OrtException {
        this.spec = spec;
        this.hidden = new float[spec.gruHidden];
        try (OrtSession.SessionOptions o = new OrtSession.SessionOptions()) {
            o.setIntraOpNumThreads(1);          // leave the other cores to Minecraft
            o.setInterOpNumThreads(1);
            o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            this.session = env.createSession(model, o);
        }
    }

    private static FloatBuffer direct(float[] a) {
        ByteBuffer b = ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.nativeOrder());
        FloatBuffer f = b.asFloatBuffer();
        f.put(a).rewind();
        return f;
    }

    @Override
    public synchronized PolicyOutput step(float[] frame, float[] state, long goalId) throws OrtException {
        int s = spec.imgSize;
        LongBuffer goal = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asLongBuffer();
        goal.put(goalId).rewind();
        try (OnnxTensor f = OnnxTensor.createTensor(env, direct(frame), new long[]{1, 3, s, s});
             OnnxTensor st = OnnxTensor.createTensor(env, direct(state), new long[]{1, state.length});
             OnnxTensor g = OnnxTensor.createTensor(env, goal, new long[]{1});
             OnnxTensor h = OnnxTensor.createTensor(env, direct(hidden), new long[]{1, 1, hidden.length});
             OrtSession.Result r = session.run(Map.of("frame", f, "state", st, "goal", g, "h", h))) {
            float[] keys = ((float[][]) r.get("keys").orElseThrow().getValue())[0];
            float[] hotbar = ((float[][]) r.get("hotbar").orElseThrow().getValue())[0];
            float[] camY = ((float[][]) r.get("cam_y").orElseThrow().getValue())[0];
            float[] camP = ((float[][]) r.get("cam_p").orElseThrow().getValue())[0];
            hidden = ((float[][][]) r.get("h_out").orElseThrow().getValue())[0][0].clone();
            return new PolicyOutput(keys, hotbar, camY, camP);
        }
    }

    @Override
    public synchronized void reset() {
        hidden = new float[spec.gruHidden];
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
        }
    }
}
