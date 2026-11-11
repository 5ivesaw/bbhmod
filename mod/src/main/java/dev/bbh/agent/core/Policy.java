package dev.bbh.agent.core;

/** One recurrent step. Implementations keep the hidden state between calls. */
public interface Policy extends AutoCloseable {
    /**
     * @param frameNchw 3*S*S floats, values 0..255, planes R,G,B
     * @param state     Spec.stateDim() floats
     * @param goalId    0 = no goal, otherwise vocab index + 1
     */
    PolicyOutput step(float[] frameNchw, float[] state, long goalId) throws Exception;

    /** Forget the hidden state (new episode, respawn, long pause). */
    void reset();

    @Override
    void close();
}
