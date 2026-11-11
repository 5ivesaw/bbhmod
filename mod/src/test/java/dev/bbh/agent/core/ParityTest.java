package dev.bbh.agent.core;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Checks the Java core against fixtures produced by the Python reference.
 * Usage: java dev.bbh.agent.core.ParityTest <parity.json> <spec.json>
 */
public final class ParityTest {
    static int failures = 0;

    static void check(boolean ok, String msg) {
        if (!ok) {
            failures++;
            System.out.println("FAIL: " + msg);
        }
    }

    static int synth(int y, int x, int c) {
        return (y * 7 + x * 13 + c * 29 + ((y * x) % 17) * 5) % 256;
    }

    public static void main(String[] args) throws Exception {
        Map<String, Object> fx = MiniJson.object(MiniJson.parse(Files.readString(Path.of(args[0]))));
        Spec spec = Spec.fromJson(Files.readString(Path.of(args[1])));

        // ---- canon ----
        Map<String, Object> canon = MiniJson.object(fx.get("canon"));
        for (Map.Entry<String, Object> e : canon.entrySet())
            check(Canon.item(e.getKey()).equals(e.getValue()), "canon " + e.getKey() + " -> " + Canon.item(e.getKey()) + " expected " + e.getValue());

        // ---- mu-law ----
        Map<String, Object> mu = MiniJson.object(fx.get("mulaw"));
        List<Object> degs = MiniJson.array(mu.get("deg")), bins = MiniJson.array(mu.get("bin")), dec = MiniJson.array(mu.get("decoded"));
        for (int i = 0; i < degs.size(); i++) {
            int got = MuLaw.encode((Double) degs.get(i), spec);
            check(got == ((Double) bins.get(i)).intValue(), "mulaw encode " + degs.get(i) + " got " + got + " expected " + bins.get(i));
        }
        double maxDec = 0;
        for (int i = 0; i < dec.size(); i++) maxDec = Math.max(maxDec, Math.abs(MuLaw.decode(i, spec) - (Double) dec.get(i)));
        check(maxDec < 1e-9, "mulaw decode max diff " + maxDec);

        // ---- area resize ----
        AreaResize rs = new AreaResize();
        int out = spec.imgSize;
        float[] dst = new float[3 * out * out];
        for (Object o : MiniJson.array(fx.get("resize"))) {
            Map<String, Object> c = MiniJson.object(o);
            int h = ((Double) c.get("h")).intValue(), w = ((Double) c.get("w")).intValue();
            int[] px = new int[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    px[y * w + x] = 0xFF000000 | (synth(y, x, 0) << 16) | (synth(y, x, 1) << 8) | synth(y, x, 2);
            rs.resize(px, w, h, out, AreaResize.Order.ARGB, dst);
            List<Object> exp = MiniJson.array(c.get("out"));     // HWC order
            int worst = 0;
            for (int oy = 0; oy < out; oy++)
                for (int ox = 0; ox < out; ox++)
                    for (int ch = 0; ch < 3; ch++) {
                        int e = ((Double) exp.get((oy * out + ox) * 3 + ch)).intValue();
                        int g = (int) dst[ch * out * out + oy * out + ox];
                        worst = Math.max(worst, Math.abs(e - g));
                    }
            check(worst <= 1, "resize " + w + "x" + h + " worst diff " + worst);
            System.out.println("resize " + w + "x" + h + " -> " + out + ": worst diff " + worst);

            // ABGR layout must give the same planes when the packed int is built the other way round
            int[] pxb = new int[w * h];
            for (int i = 0; i < px.length; i++) {
                int p = px[i];
                pxb[i] = 0xFF000000 | ((p & 255) << 16) | (p & 0xFF00) | ((p >> 16) & 255);
            }
            float[] dst2 = new float[dst.length];
            rs.resize(pxb, w, h, out, AreaResize.Order.ABGR, dst2);
            check(java.util.Arrays.equals(dst, dst2), "ABGR vs ARGB mismatch for " + w + "x" + h);
        }

        // ---- state vector ----
        Vocab vocab = new Vocab(strings(MiniJson.array(fx.get("state_vocab"))));
        for (Object o : MiniJson.array(fx.get("state"))) {
            Map<String, Object> c = MiniJson.object(o);
            Map<String, Integer> inv = new HashMap<>();
            for (Map.Entry<String, Object> e : MiniJson.object(c.get("inv")).entrySet())
                inv.put(e.getKey(), ((Double) e.getValue()).intValue());
            Snapshot s = new Snapshot(((Double) c.get("hotbar")).intValue(), (Double) c.get("pitch"),
                    (Double) c.get("yaw"), (Boolean) c.get("gui"), inv);
            float[] v = new float[spec.stateDim()];
            StateEncoder.encode(spec, vocab, s, v);
            List<Object> exp = MiniJson.array(c.get("out"));
            check(exp.size() == v.length, "state length " + exp.size() + " vs " + v.length);
            double worst = 0;
            for (int i = 0; i < v.length; i++) worst = Math.max(worst, Math.abs(v[i] - (Double) exp.get(i)));
            check(worst < 1e-6, "state worst diff " + worst);
        }

        // ---- decoder behaviour ----
        ActionDecoder d = new ActionDecoder(spec, 7);
        d.greedy = true;
        float[] keys = new float[spec.keys.size()];
        java.util.Arrays.fill(keys, -5f);
        keys[0] = 5f;
        keys[7] = 3f;
        float[] hot = new float[spec.hotbarClasses];
        hot[3] = 4f;
        float[] cy = new float[spec.camBins], cp = new float[spec.camBins];
        cy[spec.camBins - 1] = 9f;
        cp[0] = 9f;
        Action a = d.decode(new PolicyOutput(keys, hot, cy, cp));
        check(a.keys[0] && a.keys[7] && !a.keys[1], "greedy keys");
        check(a.hotbar == 3, "hotbar argmax");
        check(Math.abs(a.yawDeg - spec.camMaxDeg) < 1e-9 && Math.abs(a.pitchDeg + spec.camMaxDeg) < 1e-9, "camera extremes " + a.yawDeg + " " + a.pitchDeg);

        d.greedy = false;
        float[] flat = new float[spec.camBins];
        int[] hist = new int[spec.camBins];
        for (int i = 0; i < 20000; i++) {
            float[] centre = flat.clone();
            centre[spec.camBins / 2] = 3f;
            Action s = d.decode(new PolicyOutput(keys, hot, centre, centre));
            int idx = MuLaw.encode(s.yawDeg, spec);
            hist[idx]++;
        }
        double pCentre = Math.exp(3) / (Math.exp(3) + spec.camBins - 1);
        double got = hist[spec.camBins / 2] / 20000.0;
        check(Math.abs(got - pCentre) < 0.02, "sampling frequency " + got + " vs " + pCentre);

        if (failures == 0) System.out.println("ALL JAVA PARITY CHECKS PASSED");
        else {
            System.out.println(failures + " FAILURES");
            System.exit(1);
        }
    }

    static List<String> strings(List<Object> l) {
        java.util.ArrayList<String> r = new java.util.ArrayList<>();
        for (Object o : l) r.add((String) o);
        return r;
    }
}
