package dev.bbh.agent;

import dev.bbh.agent.core.Action;
import dev.bbh.agent.core.ActionDecoder;
import dev.bbh.agent.core.GoalBlocks;
import dev.bbh.agent.core.Policy;
import dev.bbh.agent.core.PolicyOutput;
import dev.bbh.agent.core.Snapshot;
import dev.bbh.agent.core.Spec;
import dev.bbh.agent.core.StateEncoder;
import dev.bbh.agent.core.Vocab;

import java.util.ArrayDeque;

/**
 * Owns the input devices. Exactly one owner at a time:
 *   AGENT    - the learned policy presses keys, moves the camera, picks hotbar slots.
 *   BARITONE - Baritone is executing a delegated task; the policy is paused and its memory is reset afterwards.
 *
 * Modes:
 *   LEARNED  - policy only.
 *   HYBRID   - policy, but if it makes no progress toward the goal for stuckSeconds, hand a bounded
 *              mining task to Baritone (only for goals that map to blocks in goal_blocks.json).
 *   BARITONE - Baritone does everything it can for the goal; the policy is only used when there is no
 *              block mapping for the goal.
 */
public final class AgentController {
    public enum Mode { LEARNED, HYBRID, BARITONE }
    public enum Owner { AGENT, BARITONE }

    private final MinecraftBridge bridge;
    private final Spec spec;
    private final Vocab vocab;
    private final GoalBlocks goalBlocks;
    private final Policy policy;
    private final ActionDecoder decoder;
    private final FrameGrabber frames;
    private final BaritoneAdapter baritone;
    private final float[] stateBuf;

    private volatile boolean running;
    private volatile Mode mode = Mode.HYBRID;
    private volatile Owner owner = Owner.AGENT;
    private volatile int goalId = 0;

    private int tickInStep;
    private Action current;
    private boolean paused;
    private long tick;

    // stuck detection: position every second
    private final ArrayDeque<double[]> trail = new ArrayDeque<>();
    public volatile int stuckSeconds = 8;
    public volatile double stuckRadius = 1.5;
    public volatile int delegateSeconds = 20;
    private long delegateStartTick;
    private int goalCountAtDelegate;

    // statistics
    private long ticksAgent, ticksBaritone, steps, nanosInStep;
    private String stopReason = "";

    public AgentController(MinecraftBridge bridge, Spec spec, Vocab vocab, GoalBlocks goalBlocks, Policy policy,
                           FrameGrabber frames, BaritoneAdapter baritone, long seed) {
        this.bridge = bridge;
        this.spec = spec;
        this.vocab = vocab;
        this.goalBlocks = goalBlocks;
        this.policy = policy;
        this.frames = frames;
        this.baritone = baritone;
        this.decoder = new ActionDecoder(spec, seed);
        this.stateBuf = new float[spec.stateDim()];
    }

    // ---- control surface used by commands ------------------------------------------------

    public void start() {
        policy.reset();
        frames.invalidate();
        trail.clear();
        tickInStep = 0;
        current = null;
        paused = false;
        owner = Owner.AGENT;
        ticksAgent = ticksBaritone = steps = nanosInStep = 0;
        stopReason = "";
        running = true;
    }

    public void stop(String why) {
        if (running && owner == Owner.BARITONE) baritone.cancel();
        running = false;
        owner = Owner.AGENT;
        stopReason = why;
        bridge.releaseAll();
    }

    public boolean running() { return running; }
    public Mode mode() { return mode; }
    public void mode(Mode m) { this.mode = m; }
    public Owner owner() { return owner; }
    public ActionDecoder decoder() { return decoder; }
    public int goalId() { return goalId; }

    /** @return false if the item is not in the model's vocabulary */
    public boolean setGoal(String item) {
        if (item == null || item.equalsIgnoreCase("none")) {
            goalId = 0;
            return true;
        }
        int id = vocab.goalId(item);
        if (id == 0) return false;
        goalId = id;
        return true;
    }

    public String goalName() {
        return goalId == 0 ? "none" : vocab.names.get(goalId - 1);
    }

    public String status() {
        long total = Math.max(1, ticksAgent + ticksBaritone);
        double ms = steps == 0 ? 0 : nanosInStep / 1e6 / steps;
        return String.format("running=%s mode=%s owner=%s goal=%s | learned %.0f%% / baritone %.0f%% of ticks | "
                        + "policy step avg %.2f ms over %d steps | %s | baritone=%s%s%s",
                running, mode, owner, goalName(), 100.0 * ticksAgent / total, 100.0 * ticksBaritone / total,
                ms, steps, frames.timing(), baritone.available() ? "yes" : "no (" + baritone.error() + ")",
                frames.lastError() == null ? "" : " | frame error: " + frames.lastError(),
                stopReason.isEmpty() ? "" : " | last stop: " + stopReason);
    }

    // ---- per-tick driver -------------------------------------------------------------------

    public void onClientTick() {
        if (!running) return;
        if (!bridge.inWorld() || bridge.playerDead()) {
            stop("left world or died");
            return;
        }
        if (bridge.guiOpen() || !bridge.mouseGrabbed()) {   // a human took over (menu, inventory, alt-tab)
            if (!paused) {
                bridge.releaseAll();
                if (owner == Owner.BARITONE) baritone.cancel();
                owner = Owner.AGENT;
                paused = true;
            }
            return;
        }
        if (paused) {
            paused = false;
            policy.reset();                // memory from before the interruption is no longer meaningful
            frames.invalidate();
            tickInStep = 0;
        }
        tick++;

        try {
            sampleTrail();
            if (owner == Owner.BARITONE) {
                tickBaritone();
                return;
            }
            if (maybeDelegate()) return;

            if (tickInStep == 0) {
                planStep();
                frames.request();          // result arrives asynchronously and is used by the next step
            }
            applyTick();
            ticksAgent++;
            tickInStep = (tickInStep + 1) % spec.stride;
        } catch (Exception e) {
            stop("error: " + e);
            Main.LOG.error("agent stopped", e);
        }
    }

    private void planStep() throws Exception {
        float[] frame = frames.latest(1000);
        if (frame == null) {                // nothing to look at yet: stand still rather than act blind
            current = Action.idle(spec.keys.size());
            frames.request();
            return;
        }
        long t0 = System.nanoTime();
        Snapshot snap = bridge.snapshot();
        StateEncoder.encode(spec, vocab, snap, stateBuf);
        PolicyOutput out = policy.step(frame, stateBuf, goalId);
        current = decoder.decode(out);
        nanosInStep += System.nanoTime() - t0;
        steps++;
    }

    private void applyTick() {
        Action a = current;
        if (a == null) return;
        bridge.setKeys(a.keys);
        if (tickInStep == 0 && a.hotbar > 0) bridge.selectHotbar(a.hotbar - 1);
        bridge.turn(a.yawDeg / spec.stride, a.pitchDeg / spec.stride);
    }

    // ---- Baritone delegation -----------------------------------------------------------------

    private void sampleTrail() {
        if (tick % 20 != 0) return;
        trail.addLast(bridge.position());
        while (trail.size() > stuckSeconds + 1) trail.removeFirst();
    }

    private int goalCount() {
        if (goalId == 0) return 0;
        return bridge.snapshot().inventory.getOrDefault(vocab.names.get(goalId - 1), 0);
    }

    private boolean stuck() {
        if (trail.size() < stuckSeconds + 1) return false;
        double[] a = trail.peekFirst(), b = trail.peekLast();
        return Math.hypot(a[0] - b[0], a[2] - b[2]) < stuckRadius;
    }

    private boolean delegate() {
        if (goalId == 0 || !baritone.available()) return false;
        String[] blocks = goalBlocks.blocksFor(vocab.names.get(goalId - 1));
        if (blocks == null || !baritone.mine(blocks)) return false;
        bridge.releaseAll();
        owner = Owner.BARITONE;
        delegateStartTick = tick;
        goalCountAtDelegate = goalCount();
        return true;
    }

    private boolean maybeDelegate() {
        if (mode == Mode.LEARNED || goalId == 0) return false;
        if (mode == Mode.BARITONE || (mode == Mode.HYBRID && stuck() && tick % 20 == 0)) {
            if (delegate()) {
                tickBaritone();
                return true;
            }
        }
        return false;
    }

    private void tickBaritone() {
        ticksBaritone++;
        boolean gained = goalCount() > goalCountAtDelegate;
        boolean timeout = mode == Mode.HYBRID && tick - delegateStartTick > delegateSeconds * 20L;
        boolean idle = tick - delegateStartTick > 40 && !baritone.mining();
        if (gained || timeout || idle) {
            baritone.cancel();
            owner = Owner.AGENT;
            policy.reset();
            frames.invalidate();
            trail.clear();
            tickInStep = 0;
        }
    }
}
