package dev.bbh.agent.core;

/** Camera quantisation, identical to mulaw_encode/mulaw_decode in py/bbh/encode.py. */
public final class MuLaw {
    private MuLaw() {}

    public static int encode(double deg, Spec s) {
        double x = Math.max(-s.camMaxDeg, Math.min(s.camMaxDeg, deg)) / s.camMaxDeg;
        double y = Math.signum(x) * Math.log1p(s.camMu * Math.abs(x)) / Math.log1p(s.camMu);
        long idx = (long) Math.floor((y + 1.0) / 2.0 * (s.camBins - 1) + 0.5);
        return (int) Math.max(0, Math.min(s.camBins - 1, idx));
    }

    public static double decode(int idx, Spec s) {
        double y = (double) idx / (s.camBins - 1) * 2.0 - 1.0;
        return Math.signum(y) * (Math.pow(1.0 + s.camMu, Math.abs(y)) - 1.0) / s.camMu * s.camMaxDeg;
    }
}
