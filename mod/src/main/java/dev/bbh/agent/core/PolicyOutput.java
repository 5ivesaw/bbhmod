package dev.bbh.agent.core;

/** Raw logits for one step. */
public final class PolicyOutput {
    public final float[] keys, hotbar, camY, camP;

    public PolicyOutput(float[] keys, float[] hotbar, float[] camY, float[] camP) {
        this.keys = keys;
        this.hotbar = hotbar;
        this.camY = camY;
        this.camP = camP;
    }
}
