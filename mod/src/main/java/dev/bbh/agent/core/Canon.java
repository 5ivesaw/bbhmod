package dev.bbh.agent.core;

import java.util.Locale;

/** Item-name canonicalisation. Must stay identical to canon_item() in py/bbh/encode.py. */
public final class Canon {
    private Canon() {}

    public static String item(String name) {
        String n = name.strip().toLowerCase(Locale.ROOT);
        int c = n.indexOf(':');
        if (c >= 0) n = n.substring(c + 1);
        if (n.equals("none") || n.equals("air") || n.isEmpty() || n.equals("empty")) return "none";
        if (n.equals("log") || n.equals("log2") || n.endsWith("_log") || n.endsWith("_stem")
                || n.endsWith("_wood") || n.endsWith("_hyphae")) return "log";
        if (n.equals("planks") || n.endsWith("_planks")) return "planks";
        if (n.equals("leaves") || n.equals("leaves2") || n.endsWith("_leaves")) return "leaves";
        if (n.equals("sapling") || n.endsWith("_sapling")) return "sapling";
        if (n.equals("dirt") || n.equals("coarse_dirt") || n.equals("rooted_dirt")) return "dirt";
        if (n.equals("sand") || n.equals("red_sand")) return "sand";
        if (n.equals("stone") || n.equals("cobblestone")) return n;
        if (n.endsWith("_wool") || n.equals("wool")) return "wool";
        return n;
    }
}
