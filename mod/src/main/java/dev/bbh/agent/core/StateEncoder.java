package dev.bbh.agent.core;

import java.util.Map;

/** Identical to encode_state() in py/bbh/encode.py. */
public final class StateEncoder {
    private StateEncoder() {}

    public static void encode(Spec spec, Vocab vocab, Snapshot s, float[] out) {
        java.util.Arrays.fill(out, 0f);
        int slots = spec.hotbarSlots;
        if (s.hotbar >= 0 && s.hotbar < slots) out[s.hotbar] = 1f;
        int o = slots;
        out[o] = (float) (s.pitch / 90.0);
        double rad = Math.toRadians(s.yaw);
        out[o + 1] = (float) Math.sin(rad);
        out[o + 2] = (float) Math.cos(rad);
        out[o + 3] = s.guiOpen ? 1f : 0f;
        int base = o + 4;
        double denom = Math.log(1.0 + spec.invCap);
        for (Map.Entry<String, Integer> e : s.inventory.entrySet()) {
            int j = vocab.indexOf(e.getKey());
            int count = e.getValue();
            if (j >= 0 && j < spec.vocabSize && count > 0) {
                out[base + j] = (float) (Math.log(1.0 + Math.min(count, spec.invCap)) / denom);
            }
        }
    }
}
