package dev.bbh.agent.core;

import java.util.HashMap;
import java.util.Map;

/**
 * Fractional-coverage area average, the same algorithm as resize_area() in py/bbh/encode.py.
 * Training frames were shrunk with it, so the mod must shrink the live framebuffer the same way.
 * Output is written straight into a float NCHW buffer holding integer values 0..255.
 */
public final class AreaResize {
    /** Channel layout of the packed int pixels coming from the framebuffer. */
    public enum Order { ARGB, ABGR }

    private static final class Weights {
        final int[] start;
        final double[][] w;

        Weights(int src, int dst) {
            start = new int[dst];
            w = new double[dst][];
            double scale = (double) src / dst;
            for (int i = 0; i < dst; i++) {
                double lo = i * scale, hi = (i + 1) * scale;
                int j0 = (int) Math.floor(lo);
                int j1 = Math.min((int) Math.ceil(hi), src);
                start[i] = j0;
                double[] row = new double[Math.max(j1 - j0, 0)];
                for (int j = j0; j < j1; j++) {
                    double cover = Math.min(hi, j + 1) - Math.max(lo, j);
                    row[j - j0] = cover > 0 ? cover / scale : 0.0;
                }
                w[i] = row;
            }
        }
    }

    private final Map<Long, Weights> cache = new HashMap<>();
    private double[] tmp = new double[0];

    private Weights weights(int src, int dst) {
        return cache.computeIfAbsent(((long) src << 32) | dst, k -> new Weights(src, dst));
    }

    /** px: srcW*srcH packed pixels, row-major from the top-left. dst: 3*out*out floats (R plane, G plane, B plane). */
    public void resize(int[] px, int srcW, int srcH, int out, Order order, float[] dst) {
        Weights wx = weights(srcW, out), wy = weights(srcH, out);
        int need = srcH * out * 3;
        if (tmp.length < need) tmp = new double[need];
        boolean argb = order == Order.ARGB;

        for (int y = 0; y < srcH; y++) {
            int row = y * srcW;
            for (int ox = 0; ox < out; ox++) {
                double r = 0, g = 0, b = 0;
                double[] w = wx.w[ox];
                int j0 = wx.start[ox];
                for (int k = 0; k < w.length; k++) {
                    int p = px[row + j0 + k];
                    int pr, pb;
                    if (argb) { pr = (p >> 16) & 255; pb = p & 255; } else { pr = p & 255; pb = (p >> 16) & 255; }
                    int pg = (p >> 8) & 255;
                    r += w[k] * pr;
                    g += w[k] * pg;
                    b += w[k] * pb;
                }
                int o = (y * out + ox) * 3;
                tmp[o] = r;
                tmp[o + 1] = g;
                tmp[o + 2] = b;
            }
        }
        int plane = out * out;
        for (int oy = 0; oy < out; oy++) {
            double[] w = wy.w[oy];
            int i0 = wy.start[oy];
            for (int ox = 0; ox < out; ox++) {
                double r = 0, g = 0, b = 0;
                for (int k = 0; k < w.length; k++) {
                    int o = ((i0 + k) * out + ox) * 3;
                    r += w[k] * tmp[o];
                    g += w[k] * tmp[o + 1];
                    b += w[k] * tmp[o + 2];
                }
                int d = oy * out + ox;
                dst[d] = clamp(r);
                dst[plane + d] = clamp(g);
                dst[2 * plane + d] = clamp(b);
            }
        }
    }

    private static float clamp(double v) {
        double r = Math.floor(v + 0.5);
        return (float) Math.max(0.0, Math.min(255.0, r));
    }
}
