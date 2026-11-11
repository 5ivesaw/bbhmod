package dev.bbh.agent;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

/**
 * Talks to Baritone through reflection so this mod compiles and runs without Baritone installed,
 * and keeps working across Baritone releases as long as its public API names stay the same
 * (BaritoneAPI.getProvider().getPrimaryBaritone() ... mineByName / setGoalAndPath / cancelEverything).
 */
public final class BaritoneAdapter {
    private Object baritone;
    private Method getMineProcess, mineByName, getCustomGoalProcess, setGoalAndPath, getPathingBehavior,
            cancelEverything, isActive;
    private Constructor<?> goalBlockCtor;
    private String error = "not initialised";

    public BaritoneAdapter() {
        try {
            Class<?> api = Class.forName("baritone.api.BaritoneAPI");
            Object provider = api.getMethod("getProvider").invoke(null);
            Class<?> providerType = Class.forName("baritone.api.IBaritoneProvider");
            baritone = providerType.getMethod("getPrimaryBaritone").invoke(provider);

            Class<?> ib = Class.forName("baritone.api.IBaritone");
            Class<?> mine = Class.forName("baritone.api.process.IMineProcess");
            Class<?> custom = Class.forName("baritone.api.process.ICustomGoalProcess");
            Class<?> pathing = Class.forName("baritone.api.behavior.IPathingBehavior");
            Class<?> process = Class.forName("baritone.api.process.IBaritoneProcess");
            Class<?> goal = Class.forName("baritone.api.pathing.goals.Goal");
            Class<?> goalBlock = Class.forName("baritone.api.pathing.goals.GoalBlock");

            getMineProcess = ib.getMethod("getMineProcess");
            mineByName = mine.getMethod("mineByName", int.class, String[].class);
            getCustomGoalProcess = ib.getMethod("getCustomGoalProcess");
            setGoalAndPath = custom.getMethod("setGoalAndPath", goal);
            getPathingBehavior = ib.getMethod("getPathingBehavior");
            cancelEverything = pathing.getMethod("cancelEverything");
            isActive = process.getMethod("isActive");
            goalBlockCtor = goalBlock.getConstructor(int.class, int.class, int.class);
            error = null;
        } catch (Throwable t) {
            baritone = null;
            error = t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    public boolean available() {
        return baritone != null;
    }

    public String error() {
        return error;
    }

    public boolean mine(String... blockNames) {
        if (!available()) return false;
        try {
            Object proc = getMineProcess.invoke(baritone);
            mineByName.invoke(proc, 0, (Object) blockNames);
            return true;
        } catch (Throwable t) {
            error = "mine failed: " + t;
            return false;
        }
    }

    public boolean gotoBlock(int x, int y, int z) {
        if (!available()) return false;
        try {
            Object proc = getCustomGoalProcess.invoke(baritone);
            setGoalAndPath.invoke(proc, goalBlockCtor.newInstance(x, y, z));
            return true;
        } catch (Throwable t) {
            error = "goto failed: " + t;
            return false;
        }
    }

    public void cancel() {
        if (!available()) return;
        try {
            cancelEverything.invoke(getPathingBehavior.invoke(baritone));
        } catch (Throwable t) {
            error = "cancel failed: " + t;
        }
    }

    /** True while the mine process still wants control. */
    public boolean mining() {
        if (!available()) return false;
        try {
            return (Boolean) isActive.invoke(getMineProcess.invoke(baritone));
        } catch (Throwable t) {
            return false;
        }
    }
}
