package dev.bbh.agent;

import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import dev.bbh.agent.core.AreaResize;
import dev.bbh.agent.core.GoalBlocks;
import dev.bbh.agent.core.Policy;
import dev.bbh.agent.core.Spec;
import dev.bbh.agent.core.Vocab;
import dev.bbh.agent.runtime.OnnxPolicy;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.network.chat.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Entry point. Model files are read from  <minecraft>/config/bbh/  (policy.onnx, spec.json, vocab.json,
 * goal_blocks.json) or, if absent there, from the jar under /bbh_model/.
 *
 * Commands:  /bbh start | stop | status | goal <item|none> | mode <learned|hybrid|baritone>
 *            /bbh greedy <true|false> | /bbh temp <keys> <camera> | /bbh channels <argb|abgr>
 *            /bbh dumpframe | /bbh goto <x> <y> <z>
 */
public final class Main implements ClientModInitializer {
    public static final Logger LOG = LoggerFactory.getLogger("bbh");

    private AgentController controller;
    private FrameGrabber frames;
    private BaritoneAdapter baritone;
    private Spec spec;
    private String loadError;

    @Override
    public void onInitializeClient() {
        try {
            load();
        } catch (Throwable t) {
            loadError = t.toString();
            LOG.error("bbh: model not loaded", t);
        }
        ClientTickEvents.END_CLIENT_TICK.register(c -> {
            if (controller != null) controller.onClientTick();
        });
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> dispatcher.register(
                ClientCommands.literal("bbh")
                        .then(ClientCommands.literal("start").executes(c -> run(c.getSource(), () -> {
                            controller.start();
                            return "started (" + controller.status() + ")";
                        })))
                        .then(ClientCommands.literal("stop").executes(c -> run(c.getSource(), () -> {
                            controller.stop("user command");
                            return "stopped";
                        })))
                        .then(ClientCommands.literal("status").executes(c -> run(c.getSource(), () -> controller.status())))
                        .then(ClientCommands.literal("goal").then(ClientCommands.argument("item", StringArgumentType.word())
                                .executes(c -> run(c.getSource(), () -> {
                                    String item = StringArgumentType.getString(c, "item");
                                    return controller.setGoal(item) ? "goal = " + controller.goalName()
                                            : "'" + item + "' is not in this model's vocabulary";
                                }))))
                        .then(ClientCommands.literal("mode").then(ClientCommands.argument("mode", StringArgumentType.word())
                                .executes(c -> run(c.getSource(), () -> {
                                    String m = StringArgumentType.getString(c, "mode").toUpperCase();
                                    controller.mode(AgentController.Mode.valueOf(m));
                                    return "mode = " + controller.mode();
                                }))))
                        .then(ClientCommands.literal("greedy").then(ClientCommands.argument("on", StringArgumentType.word())
                                .executes(c -> run(c.getSource(), () -> {
                                    controller.decoder().greedy = Boolean.parseBoolean(StringArgumentType.getString(c, "on"));
                                    return "greedy = " + controller.decoder().greedy;
                                }))))
                        .then(ClientCommands.literal("temp").then(ClientCommands.argument("keys", DoubleArgumentType.doubleArg(0.05, 5))
                                .then(ClientCommands.argument("camera", DoubleArgumentType.doubleArg(0.05, 5))
                                        .executes(c -> run(c.getSource(), () -> {
                                            controller.decoder().keyTemperature = DoubleArgumentType.getDouble(c, "keys");
                                            controller.decoder().camTemperature = DoubleArgumentType.getDouble(c, "camera");
                                            return "temperatures set";
                                        })))))
                        .then(ClientCommands.literal("channels").then(ClientCommands.argument("order", StringArgumentType.word())
                                .executes(c -> run(c.getSource(), () -> {
                                    frames.order = AreaResize.Order.valueOf(StringArgumentType.getString(c, "order").toUpperCase());
                                    return "channel order = " + frames.order;
                                }))))
                        .then(ClientCommands.literal("dumpframe").executes(c -> run(c.getSource(), this::dumpFrame)))
                        .then(ClientCommands.literal("goto")
                                .then(ClientCommands.argument("x", IntegerArgumentType.integer())
                                        .then(ClientCommands.argument("y", IntegerArgumentType.integer())
                                                .then(ClientCommands.argument("z", IntegerArgumentType.integer())
                                                        .executes(c -> run(c.getSource(), () -> baritone.gotoBlock(
                                                                IntegerArgumentType.getInteger(c, "x"),
                                                                IntegerArgumentType.getInteger(c, "y"),
                                                                IntegerArgumentType.getInteger(c, "z")) ? "baritone is pathing"
                                                                : "baritone unavailable: " + baritone.error()))))))));
    }

    private interface Job { String call() throws Exception; }

    private int run(FabricClientCommandSource src, Job job) {
        if (controller == null) {
            src.sendFeedback(Component.literal("bbh: model not loaded: " + loadError));
            return 0;
        }
        try {
            src.sendFeedback(Component.literal("bbh: " + job.call()));
            return 1;
        } catch (Exception e) {
            src.sendFeedback(Component.literal("bbh: " + e));
            return 0;
        }
    }

    private String dumpFrame() throws IOException {
        float[] f = frames.latest(10_000);
        if (f == null) return "no frame captured yet; run /bbh start first";
        int s = spec.imgSize;
        BufferedImage img = new BufferedImage(s, s, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < s; y++)
            for (int x = 0; x < s; x++) {
                int i = y * s + x;
                int r = (int) f[i], g = (int) f[s * s + i], b = (int) f[2 * s * s + i];
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        Path out = FabricLoader.getInstance().getConfigDir().resolve("bbh").resolve("frame.png");
        javax.imageio.ImageIO.write(img, "png", out.toFile());
        return "wrote " + out + " (what the network sees; if red and blue look swapped, run /bbh channels abgr)";
    }

    private void load() throws Exception {
        Path dir = FabricLoader.getInstance().getConfigDir().resolve("bbh");
        Files.createDirectories(dir);
        spec = Spec.fromJson(readText(dir, "spec.json"));
        Vocab vocab = Vocab.fromJson(readText(dir, "vocab.json"));
        GoalBlocks gb = exists(dir, "goal_blocks.json") ? GoalBlocks.fromJson(readText(dir, "goal_blocks.json")) : GoalBlocks.empty();
        byte[] model = readBytes(dir, "policy.onnx");
        Policy policy = new OnnxPolicy(model, spec);
        frames = new FrameGrabber(spec);
        baritone = new BaritoneAdapter();
        controller = new AgentController(new MinecraftBridge(), spec, vocab, gb, policy, frames, baritone,
                System.nanoTime());
        LOG.info("bbh loaded: img={} stride={} vocab={} baritone={}", spec.imgSize, spec.stride, vocab.names.size(),
                baritone.available() ? "found" : "not found (" + baritone.error() + ")");
    }

    private static boolean exists(Path dir, String name) {
        return Files.exists(dir.resolve(name)) || Main.class.getResource("/bbh_model/" + name) != null;
    }

    private static byte[] readBytes(Path dir, String name) throws IOException {
        Path p = dir.resolve(name);
        if (Files.exists(p)) return Files.readAllBytes(p);
        try (InputStream in = Main.class.getResourceAsStream("/bbh_model/" + name)) {
            if (in == null) throw new IOException(name + " not found in " + dir + " or in the mod jar");
            return in.readAllBytes();
        }
    }

    private static String readText(Path dir, String name) throws IOException {
        return new String(readBytes(dir, name), StandardCharsets.UTF_8);
    }
}
