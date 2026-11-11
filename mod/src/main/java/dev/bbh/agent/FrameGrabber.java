package dev.bbh.agent;

import com.mojang.blaze3d.platform.NativeImage;
import dev.bbh.agent.core.AreaResize;
import dev.bbh.agent.core.Spec;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Grabs the game's own framebuffer (HUD included, like the training videos) without rendering anything extra.
 * The GPU read-back is asynchronous, the shrink runs on a worker thread, and the policy always uses the newest
 * finished frame, so the agent sees the world about one model step (~100 ms) late, like a human reaction time.
 */
public final class FrameGrabber implements AutoCloseable {
    private final Spec spec;
    private final AreaResize resizer = new AreaResize();
    private final AtomicBoolean inFlight = new AtomicBoolean(false);
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "bbh-frame");
        t.setDaemon(true);
        return t;
    });

    private volatile float[] latest;
    private volatile long latestNanos;
    private volatile String lastError;
    /** Exponential moving averages in milliseconds: copy on the game thread, shrink on the worker. */
    private volatile double copyMs, shrinkMs;
    /** If /bbh dumpframe shows swapped red and blue, flip this with /bbh channels. */
    public volatile AreaResize.Order order = AreaResize.Order.ARGB;

    public FrameGrabber(Spec spec) {
        this.spec = spec;
    }

    public void request() {
        if (!inFlight.compareAndSet(false, true)) return;
        try {
            Screenshot.takeScreenshot(Minecraft.getInstance().getMainRenderTarget(), this::onImage);
        } catch (Throwable t) {
            lastError = t.toString();
            inFlight.set(false);
        }
    }

    private void onImage(NativeImage image) {
        int w, h;
        int[] px;
        long c0 = System.nanoTime();
        try {
            w = image.getWidth();
            h = image.getHeight();
            px = new int[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) px[y * w + x] = image.getPixel(x, y);
        } catch (Throwable t) {
            lastError = t.toString();
            inFlight.set(false);
            return;
        } finally {
            image.close();
        }
        final int fw = w, fh = h;
        copyMs = 0.9 * copyMs + 0.1 * (System.nanoTime() - c0) / 1e6;
        worker.execute(() -> {
            try {
                long s0 = System.nanoTime();
                float[] out = new float[3 * spec.imgSize * spec.imgSize];
                resizer.resize(px, fw, fh, spec.imgSize, order, out);
                shrinkMs = 0.9 * shrinkMs + 0.1 * (System.nanoTime() - s0) / 1e6;
                latest = out;
                latestNanos = System.nanoTime();
            } catch (Throwable t) {
                lastError = t.toString();
            } finally {
                inFlight.set(false);
            }
        });
    }

    /** Newest finished frame (NCHW floats 0..255) or null if none yet / too old to trust. */
    public float[] latest(long maxAgeMillis) {
        float[] f = latest;
        if (f == null) return null;
        return (System.nanoTime() - latestNanos) / 1_000_000L > maxAgeMillis ? null : f;
    }

    public void invalidate() {
        latest = null;
    }

    public String lastError() {
        return lastError;
    }

    public String timing() {
        return String.format("frame copy %.1f ms (game thread) + shrink %.1f ms (worker)", copyMs, shrinkMs);
    }

    @Override
    public void close() {
        worker.shutdownNow();
    }
}
