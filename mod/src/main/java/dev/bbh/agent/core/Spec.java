package dev.bbh.agent.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Mirror of contract/spec.json. The trained model ships its own copy, so mod and weights always agree. */
public final class Spec {
    public final int imgSize, stride, hotbarClasses, camBins, hotbarSlots, invCap, vocabSize, gruHidden, runHz;
    public final double camMaxDeg, camMu;
    public final List<String> keys;

    private Spec(Map<String, Object> m) {
        imgSize = num(m.get("img_size"));
        stride = num(m.get("stride"));
        hotbarClasses = num(m.get("hotbar_classes"));
        vocabSize = num(m.get("vocab_size"));
        gruHidden = num(m.get("gru_hidden"));
        runHz = num(m.get("run_hz"));
        Map<String, Object> cam = MiniJson.object(m.get("camera"));
        camBins = num(cam.get("bins"));
        camMaxDeg = (Double) cam.get("max_deg");
        camMu = (Double) cam.get("mu");
        Map<String, Object> st = MiniJson.object(m.get("state"));
        hotbarSlots = num(st.get("hotbar_slots"));
        invCap = num(st.get("inventory_count_cap"));
        List<String> k = new ArrayList<>();
        for (Object o : MiniJson.array(m.get("keys"))) k.add((String) o);
        keys = List.copyOf(k);
    }

    private static int num(Object o) {
        return ((Double) o).intValue();
    }

    public static Spec fromJson(String json) {
        return new Spec(MiniJson.object(MiniJson.parse(json)));
    }

    /** hotbar one-hot + pitch + sin(yaw) + cos(yaw) + gui flag + inventory vocabulary. */
    public int stateDim() {
        return hotbarSlots + 1 + 2 + 1 + vocabSize;
    }
}
