package dev.bbh.agent.core;

import java.util.Map;

/** Everything about the player the state vector needs, already in plain Java types. */
public final class Snapshot {
    public final int hotbar;
    public final double pitch, yaw;
    public final boolean guiOpen;
    /** canonical item name -> total count across the whole inventory */
    public final Map<String, Integer> inventory;

    public Snapshot(int hotbar, double pitch, double yaw, boolean guiOpen, Map<String, Integer> inventory) {
        this.hotbar = hotbar;
        this.pitch = pitch;
        this.yaw = yaw;
        this.guiOpen = guiOpen;
        this.inventory = inventory;
    }
}
