package dev.bbh.agent.core;

import java.util.SplittableRandom;

/**
 * Turns logits into an Action. Keys and camera are sampled (humans are not deterministic and
 * argmax makes the camera freeze at the centre bin); the hotbar uses argmax so slots never flicker.
 */
public final class ActionDecoder {
    private final Spec spec;
    private final SplittableRandom rng;
    public volatile double keyTemperature = 1.0;
    public volatile double camTemperature = 1.0;
    public volatile boolean greedy = false;

    public ActionDecoder(Spec spec, long seed) {
        this.spec = spec;
        this.rng = new SplittableRandom(seed);
    }

    public Action decode(PolicyOutput o) {
        int n = spec.keys.size();
        boolean[] keys = new boolean[n];
        for (int i = 0; i < n; i++) {
            double p = sigmoid(o.keys[i] / (greedy ? 1.0 : keyTemperature));
            keys[i] = greedy ? o.keys[i] > 0 : rng.nextDouble() < p;
        }
        int hot = argmax(o.hotbar);
        int cy = greedy ? argmax(o.camY) : sample(o.camY, camTemperature);
        int cp = greedy ? argmax(o.camP) : sample(o.camP, camTemperature);
        return new Action(keys, hot, MuLaw.decode(cy, spec), MuLaw.decode(cp, spec));
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    static int argmax(float[] a) {
        int b = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
        return b;
    }

    private int sample(float[] logits, double temp) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) max = Math.max(max, l);
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp((logits[i] - max) / temp);
            sum += p[i];
        }
        double r = rng.nextDouble() * sum, acc = 0;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (r < acc) return i;
        }
        return p.length - 1;
    }
}
