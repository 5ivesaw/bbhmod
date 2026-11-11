package dev.bbh.agent;

import dev.bbh.agent.core.Canon;
import dev.bbh.agent.core.Snapshot;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;

import java.util.HashMap;
import java.util.Map;

/**
 * Every call that touches Minecraft internals lives here. The rest of the mod only sees plain Java types,
 * so when a Minecraft update renames something, this is the one file to fix.
 *
 * Key order matches contract/spec.json "keys":
 * forward, back, left, right, jump, sneak, sprint, attack, use.
 */
public final class MinecraftBridge {
    /** Mouse sensitivity factor used by Entity.turn(): one "mouse unit" = 0.15 degrees. */
    private static final double DEG_PER_MOUSE_UNIT = 0.15;

    private final Minecraft mc = Minecraft.getInstance();
    private boolean prevAttack;

    public boolean inWorld() {
        return mc.player != null && mc.level != null && mc.gameMode != null;
    }

    public boolean guiOpen() {
        return mc.gui.screen() != null;
    }

    public boolean mouseGrabbed() {
        return mc.mouseHandler.isMouseGrabbed();
    }

    public boolean playerDead() {
        return mc.player == null || mc.player.isDeadOrDying();
    }

    public Snapshot snapshot() {
        LocalPlayer p = mc.player;
        Inventory inv = p.getInventory();
        Map<String, Integer> counts = new HashMap<>();
        for (int i = 0; i < inv.getContainerSize(); i++) {
            ItemStack st = inv.getItem(i);
            if (st.isEmpty()) continue;
            String name = Canon.item(BuiltInRegistries.ITEM.getKey(st.getItem()).getPath());
            counts.merge(name, st.getCount(), Integer::sum);
        }
        return new Snapshot(inv.getSelectedSlot(), p.getXRot(), p.getYRot(), mc.gui.screen() != null, counts);
    }

    public double[] position() {
        LocalPlayer p = mc.player;
        return new double[]{p.getX(), p.getY(), p.getZ()};
    }

    public void setKeys(boolean[] k) {
        var o = mc.options;
        o.keyUp.setDown(k[0]);
        o.keyDown.setDown(k[1]);
        o.keyLeft.setDown(k[2]);
        o.keyRight.setDown(k[3]);
        o.keyJump.setDown(k[4]);
        o.keyShift.setDown(k[5]);
        o.keySprint.setDown(k[6]);
        o.keyAttack.setDown(k[7]);
        o.keyUse.setDown(k[8]);

        // Vanilla already handles held keys: holding attack mines blocks and holding use places/uses items
        // (with its own 4-tick delay). The one thing it does not do from a held key is hit an ENTITY, which
        // needs a click event, so fire that on the rising edge of attack. Use is deliberately left to vanilla,
        // otherwise a block would be placed twice per press.
        HitResult hit = mc.hitResult;
        if (k[7] && !prevAttack && hit instanceof EntityHitResult e) {
            mc.gameMode.attack(mc.player, e.getEntity());
        }
        prevAttack = k[7];
    }

    public void releaseAll() {
        if (mc.options == null) return;
        setKeysQuietly();
        prevAttack = false;
    }

    private void setKeysQuietly() {
        var o = mc.options;
        o.keyUp.setDown(false);
        o.keyDown.setDown(false);
        o.keyLeft.setDown(false);
        o.keyRight.setDown(false);
        o.keyJump.setDown(false);
        o.keyShift.setDown(false);
        o.keySprint.setDown(false);
        o.keyAttack.setDown(false);
        o.keyUse.setDown(false);
    }

    public void selectHotbar(int slot0) {
        if (slot0 >= 0 && slot0 < 9) mc.player.getInventory().setSelectedSlot(slot0);
    }

    /** Rotate the camera by degrees exactly like mouse movement does (also keeps interpolation state right). */
    public void turn(double yawDeg, double pitchDeg) {
        if (yawDeg == 0.0 && pitchDeg == 0.0) return;
        mc.player.turn(yawDeg / DEG_PER_MOUSE_UNIT, pitchDeg / DEG_PER_MOUSE_UNIT);
    }

    public Minecraft minecraft() {
        return mc;
    }
}
