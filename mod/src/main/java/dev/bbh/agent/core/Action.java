package dev.bbh.agent.core;

/** Decoded decision for one model step (covers Spec.stride game ticks). */
public final class Action {
    /** same order as Spec.keys */
    public final boolean[] keys;
    /** 0 = keep slot, 1..9 = select hotbar slot (value - 1) */
    public final int hotbar;
    public final double yawDeg, pitchDeg;

    public Action(boolean[] keys, int hotbar, double yawDeg, double pitchDeg) {
        this.keys = keys;
        this.hotbar = hotbar;
        this.yawDeg = yawDeg;
        this.pitchDeg = pitchDeg;
    }

    public static Action idle(int nKeys) {
        return new Action(new boolean[nKeys], 0, 0.0, 0.0);
    }
}
